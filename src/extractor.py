from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from configs import get_model, get_task
from utils import set_seed


@dataclass
class InputExample:
    idx: int
    source: str
    label: int


@dataclass
class InputFeatures:
    idx: int
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]
    label: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature extraction")
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--task", required=True, type=str)
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_examples(
    file_path: str, task: str, dataset_name: str, split: str
) -> List[InputExample]:
    examples: List[InputExample] = []

    if task != "vul":
        raise ValueError("Only 'vul' task is supported.")

    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for idx, line in enumerate(tqdm(f, desc=f"Read {split}")):
                data = json.loads(line.strip())
                
                if dataset_name == "primevul_new":
                    examples.append(
                        InputExample(
                            idx=idx, 
                            source=data["func"],
                            label=int(data["target"])
                        )
                    )
                else:
                    raise ValueError(f"JSONL format not supported for dataset {dataset_name}")
    else:
        df = pd.read_parquet(file_path)
        
        if dataset_name == "primevul":
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Read {split}"):
                examples.append(
                    InputExample(
                        idx=idx, source=row["func"], label=1 if row["is_vulnerable"] else 0
                    )
                )
        elif dataset_name == "devign":
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Read {split}"):
                examples.append(
                    InputExample(
                        idx=idx, source=row["func"], label=1 if row["target"] else 0
                    )
                )
        elif dataset_name in {"diversevul", "diversevul_balanced"}:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Read {split}"):
                examples.append(
                    InputExample(idx=idx, source=row["func"], label=int(row["target"]))
                )
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
    
    return examples


def _prefix_tokens(model_name: str, tokenizer) -> List[str]:
    if "unixcoder" in model_name:
        return [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]
    return (
        [tokenizer.cls_token]
        if model_name
        not in {"codesage-small-v2", "codesage-base-v2", "codesage-large-v2"}
        else []
    )


def convert_examples_to_features(
    examples: List[InputExample],
    split: str,
    tokenizer,
    max_seq_length: int,
    model_name: str,
) -> List[InputFeatures]:
    feats: List[InputFeatures] = []
    lengths = []

    for ex in tqdm(examples, total=len(examples), desc=f"Tokenize {split}"):
        cand = tokenizer.tokenize(ex.source)
        lengths.append(len(cand))

        # reserve space for special tokens:
        reserve = 0
        if model_name in {
            "codesage-small-v2",
            "codesage-base-v2",
            "codesage-large-v2",
        }:
            reserve = 1  # EOS
        elif "unixcoder" in model_name:
            reserve = 4  # CLS + <encoder-only> + SEP + [SEP/EOS at end]
        else:
            reserve = 2  # CLS + SEP/EOS

        if len(cand) > max_seq_length - reserve:
            cand = cand[: max_seq_length - reserve]

        tokens = []
        tokens.extend(_prefix_tokens(model_name, tokenizer))
        tokens.extend(cand)

        if (
            model_name
            in {"codesage-small-v2", "codesage-base-v2", "codesage-large-v2"}
            or "t5" in model_name
            or model_name.lower().startswith("divot5") 
            or model_name == "random-embed"
        ):
            tokens.append(tokenizer.eos_token)
        else:
            tokens.append(tokenizer.sep_token)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention.append(0)

        feats.append(
            InputFeatures(
                idx=int(ex.idx),
                tokens=tokens,
                input_ids=input_ids,
                attention_mask=attention,
                label=int(ex.label),
            )
        )

    arr = np.array(lengths, dtype=np.int32)
    print(
        f"\n{split} stats | n={len(arr)}  max={arr.max()}  mean={arr.mean():.2f}  "
        + " ".join([f"p{p}={np.percentile(arr, p):.0f}" for p in (50, 75, 90, 95, 99)])
    )
    return feats


def write_features_jsonl(
    model,
    tokenizer,
    dataloader: DataLoader,
    features: List[InputFeatures],
    out_file: str,
    pool_cls: bool,
    pool_eos: bool,
    is_encoder_decoder: bool,
    device: torch.device,
) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        with open(out_file, "w") as w:
            for input_ids, attention_mask, idx in tqdm(
                dataloader, desc=f"Extract -> {os.path.basename(out_file)}"
            ):
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                if is_encoder_decoder:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=input_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_hidden = outputs.encoder_last_hidden_state
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden = outputs.hidden_states[-1]

                for b, ex_index in enumerate(idx):
                    f = features[ex_index.item()]
                    vecs = {}

                    if pool_cls:
                        vecs["CLS"] = last_hidden[b, 0].tolist()

                    if pool_eos:
                        eos_pos = (
                            input_ids[b] == getattr(tokenizer, "eos_token_id", -1)
                        ).nonzero(as_tuple=True)[0]
                        if len(eos_pos) > 0:
                            vecs["EOS"] = last_hidden[b, eos_pos[-1].item()].tolist()

                    mask = attention_mask[b].unsqueeze(-1)  # [L,1]
                    masked = last_hidden[b] * mask
                    denom = mask.sum(dim=0).clamp(min=1)
                    avg = (masked.sum(dim=0) / denom).tolist()

                    neg_inf = torch.finfo(last_hidden.dtype).min
                    maxed = (
                        last_hidden[b]
                        .masked_fill(mask == 0, neg_inf)
                        .max(dim=0)
                        .values.tolist()
                    )

                    vecs["AVG"] = avg
                    vecs["MAX"] = maxed

                    w.write(
                        json.dumps(
                            {
                                "index": f.idx,
                                "label": f.label,
                                "features": {
                                    k: [round(x, 12) for x in v]
                                    for k, v in vecs.items()
                                },
                            }
                        )
                        + "\n"
                    )


def main():
    args = parse_args()
    set_seed(args.seed)

    mcfg = get_model(args.model)
    tcfg = get_task(args.task, args.dataset)

    if args.model == "random-embed":
        config = mcfg["config"]()
        tokenizer = mcfg["tokenizer"]()
        model = mcfg["model"](hidden_size=768)
    else:
        cfg = mcfg["config"].from_pretrained(
            mcfg["model_path"],
            trust_remote_code=True,
            output_hidden_states=True,
            return_dict=True,
        )
        tokenizer = mcfg["tokenizer"].from_pretrained(
            mcfg["model_path"], trust_remote_code=True, config=cfg
        )
        model = mcfg["model"].from_pretrained(
            mcfg["model_path"], trust_remote_code=True, config=cfg
        )

    max_len = mcfg["max_seq_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = tcfg["base_dir"]
    paths = {
        "train": os.path.join(base, tcfg["train_file"]),
        "valid": os.path.join(base, tcfg["valid_file"]),
        "test": os.path.join(base, tcfg["test_file"]),
    }

    feats = {}
    dls = {}
    for split in ("train", "valid", "test"):
        ex = read_examples(paths[split], args.task, args.dataset, split)
        f = convert_examples_to_features(ex, split, tokenizer, max_len, args.model)
        feats[split] = f
        ds = TensorDataset(
            torch.tensor([z.input_ids for z in f], dtype=torch.long),
            torch.tensor([z.attention_mask for z in f], dtype=torch.long),
            torch.tensor([z.idx for z in f], dtype=torch.long),
        )
        dls[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    out_dir = os.path.join(base, "features", args.model)
    write_features_jsonl(
        model,
        tokenizer,
        dls["train"],
        feats["train"],
        os.path.join(out_dir, f"train_{args.model}_features.jsonl"),
        pool_cls=mcfg["pool_cls"],
        pool_eos=mcfg["pool_eos"],
        is_encoder_decoder=mcfg["is_encoder_decoder"],
        device=device,
    )
    write_features_jsonl(
        model,
        tokenizer,
        dls["valid"],
        feats["valid"],
        os.path.join(out_dir, f"valid_{args.model}_features.jsonl"),
        pool_cls=mcfg["pool_cls"],
        pool_eos=mcfg["pool_eos"],
        is_encoder_decoder=mcfg["is_encoder_decoder"],
        device=device,
    )
    write_features_jsonl(
        model,
        tokenizer,
        dls["test"],
        feats["test"],
        os.path.join(out_dir, f"test_{args.model}_features.jsonl"),
        pool_cls=mcfg["pool_cls"],
        pool_eos=mcfg["pool_eos"],
        is_encoder_decoder=mcfg["is_encoder_decoder"],
        device=device,
    )
    print(f"\nFeatures saved to {out_dir}")


if __name__ == "__main__":
    main()
