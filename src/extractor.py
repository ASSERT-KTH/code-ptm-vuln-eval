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

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction script.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., codebert, modernbert).")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., vul).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    return parser.parse_args()

def convert_examples_to_features(examples, split, tokenizer, max_seq_length):
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), desc=f"Converting {split} examples to features"):
        cand_tokens = tokenizer.tokenize(example.source)
        if len(cand_tokens) > max_seq_length - 2: 
            cand_tokens = cand_tokens[0:(max_seq_length - 2)] 

        tokens = []
        input_type_ids = []
        
        tokens.append("<s>")
        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("</s>")
        input_type_ids.append(0)

        input_ids  = tokenizer.convert_tokens_to_ids(tokens)
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
                label=example.label
            )
        )

    return features

def read_examples(file_path, task, split):
    examples = []
    df = pd.read_parquet(file_path)
    if task == "vul":
        for id, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {split} examples"):
            examples.append(
                InputExample(
                    idx=id,
                    source=row["func"],
                    label=1 if row["is_vulnerable"] else 0
                )
            )

    return examples


def generate_features_json(dataloader, features, task, split):
    full_model_name = model.config._name_or_path
    model_name = full_model_name.split("/")[-1]
    base_dir = task["base_dir"]
    output_dir = f"{base_dir}/features/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{split}_{model_name}_features.jsonl"

    with open(output_file, "w") as writer:
        # Disabling Gradient Computation
        with torch.no_grad():
            # Iterating Over the DataLoader
            for batch in tqdm(dataloader, desc=f"Extracting features for {split} split", total=len(dataloader)):
                input_ids, attention_mask, idx = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                # Forward Pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.hidden_states[-1]
                # Iterating Over Batch Indices
                for iter_index, example_index in enumerate(idx):
                    feature = features[example_index.item()]
                    unique_id = int(feature.idx)
                    label = feature.label
                    cls_features = last_hidden_state[iter_index][0].tolist()
                    avg_features = torch.mean(last_hidden_state[iter_index], dim=0).tolist()
                    max_features = torch.max(last_hidden_state[iter_index], dim=0).values.tolist()
                    output_json = {
                        "index": unique_id,
                        "label": label,
                        "features": {
                            "CLS": [round(value, 12) for value in cls_features],
                            "AVG": [round(value, 12) for value in avg_features],
                            "MAX": [round(value, 12) for value in max_features],
                        }
                    }
                    writer.write(json.dumps(output_json) + "\n")

    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load the model and task configurations
    model_config = get_model(args.model)
    task = get_task(args.task)

    # Load the model and tokenizer
    tokenizer = model_config["tokenizer"].from_pretrained(model_config["model_path"])
    model = model_config["model"].from_pretrained(
        model_config["model_path"],
        config=model_config["config"].from_pretrained(
            model_config["model_path"],
            output_hidden_states=True
        )
    )
    max_seq_length = model_config["max_seq_length"]

    # Load dataset paths
    base_dir = task["base_dir"]
    train_file_path = os.path.join(base_dir, task["train_file"])
    test_file_path = os.path.join(base_dir, task["test_file"])
    valid_file_path = os.path.join(base_dir, task["valid_file"])

    # Read examples
    train_examples = read_examples(train_file_path, args.task, "train")
    test_examples = read_examples(test_file_path, args.task, "test")
    valid_examples = read_examples(valid_file_path, args.task, "valid")

    # Convert examples to features
    train_features = convert_examples_to_features(train_examples, "train", tokenizer, max_seq_length)
    test_features = convert_examples_to_features(test_examples, "test", tokenizer, max_seq_length)
    valid_features = convert_examples_to_features(valid_examples, "valid", tokenizer, max_seq_length)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([int(f.idx) for f in train_features], dtype=torch.long),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in test_features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in test_features], dtype=torch.long),
        torch.tensor([int(f.idx) for f in test_features], dtype=torch.long),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    valid_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in valid_features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in valid_features], dtype=torch.long),
        torch.tensor([int(f.idx) for f in valid_features], dtype=torch.long),
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