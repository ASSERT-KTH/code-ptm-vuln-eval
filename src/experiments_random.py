from __future__ import annotations

import argparse
import os
from argparse import Namespace
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from classifier import run
from configs import get_task
from extractor import convert_examples_to_features, read_examples, write_features_jsonl
from random_model import DummyTokenizer, RandomEmbeddingModel
from utils import load_features_jsonl, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Random-embedding: extract on many seeds + classify on 10 seeds each"
    )
    p.add_argument("--task", required=True, type=str, help="Task name (e.g., vul)")
    p.add_argument(
        "--dataset", required=True, type=str, help="Train/val dataset (e.g., primevul)"
    )
    p.add_argument(
        "--test_dataset", type=str, help="Test dataset (defaults to --dataset)"
    )
    p.add_argument(
        "--batch_size", type=int, default=16, help="Feature extraction batch size"
    )
    p.add_argument(
        "--classifier_batch_size", type=int, default=16, help="Classifier batch size"
    )
    p.add_argument("--num_epochs", type=int, default=10, help="Classifier epochs")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Classifier LR")
    p.add_argument(
        "--method", type=str, default="CLS", choices=["CLS", "AVG", "MAX", "EOS"]
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--embed_seeds",
        type=str,
        default="",
        help="Comma-separated embedding seeds. If set, overrides --n_embed_seeds.",
    )
    p.add_argument(
        "--n_embed_seeds",
        type=int,
        default=0,
        help="Generate this many embedding seeds starting from --embed_seed_start (default 0 means skip).",
    )
    p.add_argument(
        "--embed_seed_start",
        type=int,
        default=73,
        help="Starting value to generate embedding seeds when using --n_embed_seeds.",
    )
    p.add_argument(
        "--extract",
        action="store_true",
        help="If set, (re)extract features for all embedding seeds.",
    )
    p.add_argument(
        "--classifier_seeds",
        type=str,
        default="42,123,456,789,101112,131415,161718,192021,222324,252627",
        help="Comma-separated seeds for classifier runs per embedding seed (default 10).",
    )
    p.add_argument(
        "--same_seed",
        action="store_true",
        help="Use the classifier seed equal to the embedding seed (one run per embed seed).",
    )

    return p.parse_args()


def parse_seed_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def resolve_embed_seeds(args) -> List[int]:
    expl = parse_seed_list(args.embed_seeds)
    if expl:
        return expl
    if args.n_embed_seeds > 0:
        return [args.embed_seed_start + i * 9973 for i in range(args.n_embed_seeds)]
    return [args.embed_seed_start]


def extract_random_features(
    task_cfg: dict,
    dataset_name: str,
    seed: int,
    splits: List[str],
    bs: int = 16,
    num_workers: int = 2,
):
    print(f"\n[extract][seed={seed}] ➜ starting extraction for splits={splits}")
    set_seed(seed)
    tokenizer = DummyTokenizer()
    model = RandomEmbeddingModel(hidden_size=768)
    max_len = 512
    base_dir = task_cfg["base_dir"]

    files = {
        "train": os.path.join(base_dir, task_cfg["train_file"]),
        "valid": os.path.join(base_dir, task_cfg["valid_file"]),
        "test": os.path.join(base_dir, task_cfg["test_file"]),
    }

    out_dir = os.path.join(base_dir, "features", f"random-embed-seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in splits:
        out_path = os.path.join(out_dir, f"{split}_random-embed_features.jsonl")
        if os.path.exists(out_path):
            continue

        ex = read_examples(files[split], "vul", dataset_name, split)
        feats = convert_examples_to_features(
            ex, split, tokenizer, max_len, "random-embed"
        )
        ds = TensorDataset(
            torch.tensor([f.input_ids for f in feats], dtype=torch.long),
            torch.tensor([f.attention_mask for f in feats], dtype=torch.long),
            torch.tensor([f.idx for f in feats], dtype=torch.long),
        )
        dl = DataLoader(
            ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        write_features_jsonl(
            model,
            tokenizer,
            dl,
            feats,
            out_path,
            pool_cls=True,
            pool_eos=True,
            is_encoder_decoder=False,
            device=device,
        )
        print(f"[extract][seed={seed}] split={split} wrote={out_path}")


def main():
    args = parse_args()
    train_task = get_task(args.task, args.dataset)
    test_task = (
        get_task(args.task, args.test_dataset) if args.test_dataset else train_task
    )

    embed_seeds = resolve_embed_seeds(args)
    print(f"[config] embed_seeds (count={len(embed_seeds)}): {embed_seeds}")
    clf_seeds = parse_seed_list(args.classifier_seeds)
    if not clf_seeds and not args.same_seed:
        raise ValueError(
            "No classifier seeds resolved. Pass --classifier_seeds or use --same_seed."
        )

    if args.extract:
        print(f"[extract] Embedding seeds: {embed_seeds}")
        for es in tqdm(embed_seeds, desc="extract seeds"):
            extract_random_features(
                train_task,
                args.dataset,
                es,
                ["train", "valid"],
                bs=args.batch_size,
                num_workers=args.num_workers,
            )
            extract_random_features(
                test_task,
                args.test_dataset or args.dataset,
                es,
                ["test"],
                bs=args.batch_size,
                num_workers=args.num_workers,
            )
        print("[extract] Done.")
    
    all_results: Dict[int, Dict[str, np.ndarray]] = {}
    print(f"[classify] Classifier seeds: {clf_seeds}")
    for es in tqdm(embed_seeds, desc="classify over embedding seeds"):
        print(f"\n[classify][embed_seed={es}] ➜ loading features")
        tr = load_features_jsonl(
            os.path.join(
                train_task["base_dir"],
                "features",
                f"random-embed-seed{es}",
                "train_random-embed_features.jsonl",
            ),
            args.method,
        )
        va = load_features_jsonl(
            os.path.join(
                train_task["base_dir"],
                "features",
                f"random-embed-seed{es}",
                "valid_random-embed_features.jsonl",
            ),
            args.method,
        )
        te = load_features_jsonl(
            os.path.join(
                test_task["base_dir"],
                "features",
                f"random-embed-seed{es}",
                "test_random-embed_features.jsonl",
            ),
            args.method,
        )

        current_clf_seeds = [es] if args.same_seed else clf_seeds
        per_seed = []
        for cs in tqdm(current_clf_seeds, leave=False, desc=f"embed_seed={es}"):
            set_seed(cs)
            run_args = Namespace(
                task=args.task,
                dataset=args.dataset,
                test_dataset=args.test_dataset,
                model="random-embed",
                batch_size=args.classifier_batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                method=args.method,
                wandb=False,
                num_workers=args.num_workers,
            )
            per_seed.append(run(cs, run_args, tr, va, te))

        metrics = {
            k: np.array([r["test_metrics"][k] for r in per_seed])
            for k in ("accuracy", "f1", "mcc")
        }
        all_results[es] = metrics
        n = len(current_clf_seeds)
        print(
            f"\n[embed_seed={es}] Test metrics across {n} classifier seed{'s' if n != 1 else ''}:"
        )

        def mean_and_std(arr, n_runs: int):
            arr = np.asarray(arr)
            mean = float(arr.mean()) if arr.size else float("nan")
            std = float(arr.std(ddof=1)) if n_runs >= 2 else None
            return mean, std

        for name, arr in metrics.items():
            mean, std = mean_and_std(arr, n)
            std_str = f"{std:.3f}" if std is not None else "n/a"
            print(f"  {name.upper():<9} mean±std: {mean:.3f} ± {std_str}")

    f1_means = []
    missing = []
    for es in embed_seeds:
        if es in all_results:
            f1_means.append(float(all_results[es]["f1"].mean()))
        else:
            missing.append(es)

    print("\n[overall] Per-embedding-seed mean F1 (ordered by embed_seeds):")
    print("  count:", len(f1_means))
    if missing:
        print("  (warning) missing seeds with no results:", missing)

    print("  values:", [f"{v:.6f}" for v in f1_means])


if __name__ == "__main__":
    main()
