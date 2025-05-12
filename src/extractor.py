import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch
import collections
import json
import os
import argparse
from configs import get_model, get_task
import numpy as np
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
import random
import numpy as np

@dataclass
class InputExample:
    idx: str
    source: str
    label: int

@dataclass
class InputFeatures:
    idx: str
    tokens: list
    input_ids: list
    attention_mask: list
    label: int

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction script.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., codebert, modernbert).")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., vul).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., devign, primevul).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    return parser.parse_args()

def convert_examples_to_features(examples, split, tokenizer, max_seq_length):
    features = []
    exceeded_count = 0
    total_count = len(examples)
    max_length_seen = 0
    length_distribution = []
    print(args.model)
    for ex_index, example in tqdm(
        enumerate(examples),
        total=len(examples),
        desc=f"Converting {split} examples to features",
    ):
        cand_tokens = tokenizer.tokenize(example.source)
        original_length = len(cand_tokens)
        length_distribution.append(original_length)
        max_length_seen = max(max_length_seen, original_length)
        
        if original_length > max_seq_length:
            exceeded_count += 1

        if args.model == "codesage":
            if len(cand_tokens) > max_seq_length - 1:
                cand_tokens = cand_tokens[0 : (max_seq_length - 1)]
        elif args.model == "unixcoder":
            if len(cand_tokens) > max_seq_length - 4:
                cand_tokens = cand_tokens[0 : (max_seq_length - 4)]
        else:
            if len(cand_tokens) > max_seq_length - 2:
                cand_tokens = cand_tokens[0 : (max_seq_length - 2)]

        tokens = []
        input_type_ids = []

        if args.model != "codesage":
            tokens.append(tokenizer.cls_token)
            input_type_ids.append(0)

        if args.model == "unixcoder":
            tokens.append("<encoder-only>")
            input_type_ids.append(0)
            tokens.append(tokenizer.sep_token)
            input_type_ids.append(0)
        
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)

        if args.model == "codesage" or args.model == "codet5" or args.model == "codet5+" or args.model == "codit5" or args.model == "ast-t5" or args.model == "divot5":
            tokens.append(tokenizer.eos_token)
            input_type_ids.append(0)
        else:
            tokens.append(tokenizer.sep_token)
            input_type_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_type_ids) == max_seq_length
        features.append(
            InputFeatures(
                idx=example.idx,
                tokens=tokens,
                input_ids=input_ids,
                attention_mask=input_mask,
                label=example.label,
            )
        )

    print(f"\nStatistics for {split} split:")
    print(f"Total examples: {total_count}")
    print(f"Examples exceeding max length ({max_seq_length}): {exceeded_count} ({(exceeded_count/total_count)*100:.2f}%)")
    print(f"Maximum length seen: {max_length_seen}")
    print(f"Average length: {sum(length_distribution)/len(length_distribution):.2f}")
    
    percentiles = [25, 50, 75, 90, 95, 99]
    length_array = np.array(length_distribution)
    for p in percentiles:
        value = np.percentile(length_array, p)
        print(f"{p}th percentile: {value:.2f}")
    return features


def read_examples(file_path, task, dataset_name, split):
    examples = []
    df = pd.read_parquet(file_path)
    if task == "vul":
        if dataset_name == "primevul":
            for id, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {split} examples"):
                examples.append(
                    InputExample(
                        idx=id,
                        source=row["func"],
                        label=1 if row["is_vulnerable"] else 0
                    )
                )
        elif dataset_name == "devign":
            for id, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {split} examples"):
                examples.append(
                    InputExample(
                        idx=id,
                        source=row["func"],
                        label=1 if row["target"] else 0
                    )
                )
        elif dataset_name == "diversevul" or dataset_name == "diversevul_balanced":
            for id, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {split} examples"):
                examples.append(
                    InputExample(
                        idx=id,
                        source=row["func"],
                        label=row["target"]
                    )
                )
    return examples


def generate_features_json(dataloader, features, task, split):
    full_model_name = model.config._name_or_path
    model_name = full_model_name.split("/")[-1].lower()
    if args.model =="codit5":
        model_name = "codit5"
    base_dir = task["base_dir"]
    output_dir = f"{base_dir}/features/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{split}_{model_name}_features.jsonl"

    has_cls = any(m in args.model for m in MODELS_WITH_CLS)
    has_eos = any(m in args.model for m in MODELS_WITH_EOS)
    print(f"Model has CLS token: {has_cls}, EOS token: {has_eos}")

    with open(output_file, "w") as writer:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting features for {split} split", total=len(dataloader)):
                input_ids, attention_mask, idx = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                if "codet5" in model.__dict__["config"]._name_or_path or "ast_t5" in model.__dict__["config"]._name_or_path or args.model == "codit5" or args.model == "divot5":
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=input_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_hidden_state = outputs.encoder_last_hidden_state
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs.hidden_states[-1]

                for iter_index, example_index in enumerate(idx):
                    feature = features[example_index.item()]
                    unique_id = int(feature.idx)
                    label = feature.label
                    feature_dict = {}

                    if has_cls:
                        cls_features = last_hidden_state[iter_index][0].tolist()
                        feature_dict["CLS"] = [round(value, 12) for value in cls_features]

                    if has_eos:
                        eos_mask = (input_ids[iter_index] == tokenizer.eos_token_id)
                        eos_indices = eos_mask.nonzero(as_tuple=True)[0]
                        if len(eos_indices) > 0:
                            eos_index = eos_indices[-1].item()
                            eos_features = last_hidden_state[iter_index][eos_index].tolist()
                            feature_dict["EOS"] = [round(value, 12) for value in eos_features]

                    avg_features = (last_hidden_state[iter_index] * attention_mask[iter_index].unsqueeze(-1)).sum(dim=0) / attention_mask[iter_index].sum(dim=0, keepdim=True)

                    
                    mask = attention_mask[iter_index].unsqueeze(-1)
                    neg_inf = torch.finfo(last_hidden_state.dtype).min
                    masked_hs = last_hidden_state[iter_index].masked_fill(mask == 0, neg_inf)

                    max_features = masked_hs.max(dim=0).values


                    avg_features = avg_features.tolist()
                    max_features = max_features.tolist()

                    

                    feature_dict["AVG"] = [round(value, 12) for value in avg_features]
                    feature_dict["MAX"] = [round(value, 12) for value in max_features]

                    output_json = {
                        "index": unique_id,
                        "label": label,
                        "features": feature_dict
                    }
                    writer.write(json.dumps(output_json) + "\n")

    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    args = parse_args()

    model_config = get_model(args.model)
    task = get_task(args.task, args.dataset)
    dataset_name = args.dataset

    if args.model == "codet5" or args.model == "codet5+" or args.model == "codit5" or args.model == "ast-t5":
        config = model_config["config"].from_pretrained(
            model_config["model_path"],
            trust_remote_code=True,
            output_hidden_states=True,
            return_dict=True
        )
    else:
        config = model_config["config"].from_pretrained(
            model_config["model_path"],
            trust_remote_code=True,
            output_hidden_states=True,
        )
    tokenizer = model_config["tokenizer"].from_pretrained(
        model_config["model_path"],
        trust_remote_code=True,
        config=config,
    )
    model = model_config["model"].from_pretrained(
        model_config["model_path"],
        trust_remote_code=True,
        config=config,
    )
    max_seq_length = model_config["max_seq_length"]
    MODELS_WITH_CLS = ["codebert", "graphcodebert", "modernbert", "unixcoder", "codet5", "codet5+", "codit5", "ast-t5", "divot5"]
    MODELS_WITH_EOS = ["codet5", "codet5+", "codit5", "ast-t5", "divot5"]

    # Load dataset paths
    base_dir = task["base_dir"]

    train_file_path = os.path.join(base_dir, task["train_file"])
    test_file_path = os.path.join(base_dir, task["test_file"])
    valid_file_path = os.path.join(base_dir, task["valid_file"])

    # Read examples
    train_examples = read_examples(train_file_path, args.task, dataset_name, "train")
    test_examples = read_examples(test_file_path, args.task, dataset_name, "test")
    valid_examples = read_examples(valid_file_path, args.task, dataset_name, "valid")

    # # Convert examples to features
    train_features = convert_examples_to_features(train_examples, "train", tokenizer, max_seq_length)
    test_features = convert_examples_to_features(test_examples, "test", tokenizer, max_seq_length)
    valid_features = convert_examples_to_features(valid_examples, "valid", tokenizer, max_seq_length)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.idx for f in train_features], dtype=torch.long),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in test_features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in test_features], dtype=torch.long),
        torch.tensor([f.idx for f in test_features], dtype=torch.long),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    valid_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in valid_features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in valid_features], dtype=torch.long),
        torch.tensor([f.idx for f in valid_features], dtype=torch.long),
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Set device and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Extract features and save to JSON
    generate_features_json(train_dataloader, train_features, task, "train")
    generate_features_json(test_dataloader, test_features, task, "test")
    generate_features_json(valid_dataloader, valid_features, task, "valid")