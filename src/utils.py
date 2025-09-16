from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def build_loaders(
    train: Tuple[np.ndarray, np.ndarray],
    valid: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _dl(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return _dl(*train, True), _dl(*valid, False), _dl(*test, False)


@dataclass
class LoadedFeatures:
    X: np.ndarray
    y: np.ndarray
    feat_dim: int
    cat2id: Dict[str, int]
    id2cat: Dict[int, str]


def load_features_jsonl(path: str, method: str) -> LoadedFeatures:
    features, labels = [], []
    cat2id: Dict[str, int] = {}
    id2cat: Dict[int, str] = {}

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if method not in data["features"]:
                raise ValueError(f"Pooling method '{method}' not in file {path}.")
            features.append(data["features"][method])
            label_name = data["label"]
            if label_name not in cat2id:
                cat_id = len(cat2id)
                cat2id[label_name] = cat_id
                id2cat[cat_id] = label_name
            labels.append(cat2id[label_name])

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return LoadedFeatures(X=X, y=y, feat_dim=X.shape[1], cat2id=cat2id, id2cat=id2cat)


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
