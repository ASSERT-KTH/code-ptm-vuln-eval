from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from configs import get_model, get_task
from utils import (
    LoadedFeatures,
    build_loaders,
    load_features_jsonl,
    metrics_from_preds,
    set_seed,
)

try:
    import wandb
except Exception:
    wandb = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate a classifier on extracted features."
    )
    p.add_argument("--task", type=str, required=True, help="Task name (e.g., vul).")
    p.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., codebert-base)."
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name for train/val (e.g., primevul).",
    )
    p.add_argument(
        "--test_dataset",
        type=str,
        required=False,
        help="Dataset name for test (e.g., diversevul).",
    )
    p.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for DataLoader."
    )
    p.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="CLS",
        choices=["CLS", "AVG", "MAX", "EOS"],
        help="Method to use for feature extraction.",
    )
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="'train' trains the classifier and saves checkpoints; 'eval' loads saved checkpoints and evaluates on the test set.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(eval mode only) Path to a single checkpoint .pt file. "
        "When set, only that checkpoint is evaluated instead of all 10 seeds.",
    )
    return p.parse_args()


def get_feature_file(base_dir: str, model_name: str, split: str) -> str:
    return os.path.join(
        base_dir, "features", model_name, f"{split}_{model_name}_features.jsonl"
    )


class Classifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int | None = None,
        pdrop: float = 0.2,
    ):
        super().__init__()
        hidden = hidden_size or max(128, input_size // 2)
        self.net = nn.Sequential(
            nn.Dropout(pdrop),
            nn.Linear(input_size, hidden),
            nn.Tanh(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Trainer:
    def __init__(self, model: nn.Module, lr: float, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.crit = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def _step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = (b.to(self.device, non_blocking=True) for b in batch)
        logits = self.model(x)
        loss = self.crit(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        losses, y_true, y_pred = [], [], []
        for batch in loader:
            self.opt.zero_grad(set_to_none=True)
            loss, preds, y = self._step(batch)
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
        m = metrics_from_preds(np.array(y_true), np.array(y_pred))
        m["loss"] = float(np.mean(losses))
        return m

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for batch in loader:
                loss, preds, y = self._step(batch)
                losses.append(loss.item())
                y_true.extend(y.detach().cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())
        m = metrics_from_preds(np.array(y_true), np.array(y_pred))
        m["loss"] = float(np.mean(losses))
        m["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        return m


def run(
    seed: int,
    args: argparse.Namespace,
    train: LoadedFeatures,
    valid: LoadedFeatures,
) -> Dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = build_loaders(
        (train.X, train.y),
        (valid.X, valid.y),
        (valid.X, valid.y),  # dummy placeholder — test is not used during training
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    clf = Classifier(input_size=train.feat_dim, num_classes=len(train.cat2id))
    trainer = Trainer(clf, lr=args.learning_rate, device=device)

    run_name = f"{args.task}_{args.dataset}_{args.model}_{args.method}_seed{seed}"
    if args.wandb and wandb is not None:
        wandb.init(
            project="code-llm-embedding-eval",
            name=run_name,
            config=dict(vars(args), seed=seed),
        )

    best_f1, best_state = -1.0, None
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

    for epoch in range(1, args.num_epochs + 1):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.evaluate(val_loader)
        if args.wandb and wandb is not None:
            wandb.log(
                {f"train/{k}": v for k, v in train_m.items() if k != "confusion_matrix"}
            )
            wandb.log(
                {f"val/{k}": v for k, v in val_m.items() if k != "confusion_matrix"}
            )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_m['loss']:.4f} f1 {train_m['f1']:.4f} | "
            f"val loss {val_m['loss']:.4f} f1 {val_m['f1']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = {
                "epoch": epoch,
                "model_state_dict": clf.state_dict(),
                "optimizer_state_dict": trainer.opt.state_dict(),
                "val_f1": val_m["f1"],
                "val_accuracy": val_m["accuracy"],
                "val_mcc": val_m["mcc"],
                "feat_dim": train.feat_dim,
                "num_classes": len(train.cat2id),
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "method": args.method,
                },
            }
            torch.save(best_state, ckpt_path)
            print(f"   saved best checkpoint to {ckpt_path}")

    if args.wandb and wandb is not None:
        wandb.finish()

    return {"model_path": ckpt_path, "val_f1": best_f1}


def eval_run(
    seed: int,
    args: argparse.Namespace,
    test: LoadedFeatures,
    ckpt_path: str,
) -> Dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    feat_dim = state["feat_dim"]
    num_classes = state["num_classes"]

    ds = TensorDataset(
        torch.tensor(test.X, dtype=torch.float32),
        torch.tensor(test.y, dtype=torch.long),
    )
    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    clf = Classifier(input_size=feat_dim, num_classes=num_classes)
    clf.load_state_dict(state["model_state_dict"])
    trainer = Trainer(clf, lr=state["hyperparameters"]["learning_rate"], device=device)

    test_m = trainer.evaluate(test_loader)
    seed_label = f"Seed {seed}" if seed is not None else os.path.basename(ckpt_path)
    print(f"\n{seed_label} — Test results")
    print(test_m["confusion_matrix"])
    print({k: v for k, v in test_m.items() if k != "confusion_matrix"})

    return {"val_f1": state["val_f1"], "test_metrics": test_m}


if __name__ == "__main__":
    args = parse_args()

    train_task = get_task(args.task, args.dataset)
    test_task = (
        get_task(args.task, args.test_dataset) if args.test_dataset else train_task
    )

    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]

    if args.mode == "train":
        train_file = get_feature_file(train_task["base_dir"], args.model, "train")
        valid_file = get_feature_file(train_task["base_dir"], args.model, "valid")
        train = load_features_jsonl(train_file, args.method)
        valid = load_features_jsonl(valid_file, args.method)

        results = [run(seed, args, train, valid) for seed in seeds]

        best = max(results, key=lambda r: r["val_f1"])
        print(f"\nBest validation F1: {best['val_f1']:.4f}")
        print(f"Checkpoints saved in: checkpoints/")

    elif args.mode == "eval":
        test_file = get_feature_file(test_task["base_dir"], args.model, "test")
        test = load_features_jsonl(test_file, args.method)

        if args.checkpoint:
            # Single-checkpoint evaluation
            result = eval_run(
                seed=None, args=args, test=test, ckpt_path=args.checkpoint
            )
            print(f"\nTest dataset: {args.test_dataset or args.dataset}")
            print(f"Pooling: {args.method} | Model: {args.model}")
            for name in ("accuracy", "f1", "mcc"):
                print(f"{name.upper():<9}: {result['test_metrics'][name]:.6f}")
        else:
            # Full 10-seed evaluation
            results = []
            for seed in seeds:
                run_name = (
                    f"{args.task}_{args.dataset}_{args.model}_{args.method}_seed{seed}"
                )
                ckpt_path = os.path.join("checkpoints", f"{run_name}_best.pt")
                results.append(eval_run(seed, args, test, ckpt_path))

            metrics = {
                k: np.array([r["test_metrics"][k] for r in results])
                for k in ("accuracy", "f1", "mcc")
            }
            print(
                f"\nDataset: {args.dataset} | Test dataset: {args.test_dataset or args.dataset}"
            )
            print(f"Pooling: {args.method} | Model: {args.model}")

            for name, arr in metrics.items():
                print(
                    f"{name.upper():<9} mean±std: {arr.mean():.6f} ± {arr.std(ddof=1):.6f}  values: {[f'{v:.6f}' for v in arr]}"
                )
