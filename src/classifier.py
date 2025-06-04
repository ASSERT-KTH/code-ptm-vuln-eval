import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from configs import get_model, get_task
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
import wandb
from datetime import datetime
import random
import numpy as np
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier on extracted features.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., vul).")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., codebert).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name for train/val (e.g., primevul).")
    parser.add_argument("--test_dataset", type=str, required=False, help="Dataset name for test (e.g., diversevul).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--method", type=str, default="CLS", choices=["CLS", "AVG", "MAX", "EOS"], help="Method to use for feature extraction.")

    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(file_path, method="CLS"):
    features = []
    labels = []
    cat2id = {}
    id2cat = {}

    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)
    with open(file_path, "r") as f:
        for line in tqdm(f, desc=f"Loading data from {file_path}", total=total_lines):
            data = json.loads(line)
            if method not in data["features"]:
                raise ValueError(f"Pooling method '{method}' not found in features.")
            feature_vector = data["features"][method]
            features.append(feature_vector)
            label = data["label"]
            if label not in cat2id:
                cat2id[label] = len(cat2id)
                id2cat[cat2id[label]] = label
            labels.append(cat2id[label])

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    feat_dim = X.shape[1]
    return X, y, feat_dim, cat2id, id2cat

def get_file_path(task_config, split):
    base_dir = task_config["base_dir"]
    model_name = args.model
    return f"{base_dir}/features/{model_name}/{split}_{model_name}_features.jsonl"


def run_experiment(seed: int, model, train_loader, val_loader, test_loader, args) -> Dict:
    
    run_name = f"{args.task}_{args.dataset}_{args.model}_{args.method}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="code-llm-embedding-eval",
        name=run_name,
        config={
            "task": args.task,
            "dataset": args.dataset,
            "model": args.model,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "method": args.method,
            "seed": seed
        }
    )

    model_path = train_model(model, train_loader, val_loader, args.num_epochs, args.learning_rate)
    checkpoint = torch.load(model_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_model(model, test_loader)
    
    wandb.finish()
    
    return {
        'model_path': model_path,
        'val_f1': checkpoint['val_f1'],
        'test_metrics': test_metrics
    }


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    mcc = matthews_corrcoef(all_labels, all_preds)

    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print("=" * 50)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc
    }

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=768//2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_f1 = 0.0
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = f"{args.task}_{args.dataset}_{args.model}_{args.method}_lr{learning_rate}"
    best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            train_loss += loss.item()

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate epoch-level metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_loss /= len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_mcc = matthews_corrcoef(all_labels, all_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'val_mcc': val_mcc,
                'hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': args.batch_size,
                    'method': args.method,
                }
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with validation F1: {val_f1:.4f} to {best_model_path}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "train_mcc": train_mcc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_mcc": val_mcc,
        })

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
            f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, "
            f"F1 Score: {train_f1:.4f}, MCC: {train_mcc:.4f}"
        )
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
            f"Validation F1 Score: {val_f1:.4f}, Validation MCC: {val_mcc:.4f}"
        )
    return best_model_path


def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            predicted = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_mcc = matthews_corrcoef(all_labels, all_preds)

    # wandb.log({
    #     "test_accuracy": test_accuracy,
    #     "test_f1": test_f1,
    #     "test_mcc": test_mcc,
    # })
    print(
        f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, "
        f"Test MCC: {test_mcc:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()

    trainval_task = get_task(args.task, args.dataset)
    model_config = get_model(args.model)

    if args.test_dataset:
        test_task = get_task(args.task, args.test_dataset)
    else:
        test_task = trainval_task

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    method = args.method


    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
    results = []
    
    for seed in seeds:
        set_seed(seed)

        train_file = get_file_path(trainval_task, "train")
        valid_file = get_file_path(trainval_task, "valid")
        test_file = get_file_path(test_task, "test")

        train_X, train_y, feat_dim, cat2id, id2cat = load_data(train_file, method)

        valid_X, valid_y, _, _, _ = load_data(valid_file, method)

        test_X, test_y, _, _, _ = load_data(test_file, method)


        # Convert NumPy arrays to PyTorch tensors
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.long)
        valid_X = torch.tensor(valid_X, dtype=torch.float32)
        valid_y = torch.tensor(valid_y, dtype=torch.long)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.long)

        # Create dataset objects
        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        val_dataset = torch.utils.data.TensorDataset(valid_X, valid_y)
        test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nRunning experiment with seed {seed}")
        model = Classifier(input_size=feat_dim, num_classes=len(cat2id))
        result = run_experiment(
            seed=seed,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args
        )
        results.append(result)
    
    # Find best model based on validation F1
    best_result = max(results, key=lambda x: x['val_f1'])
    print(f"\nBest model validation F1: {best_result['val_f1']:.4f}")
    
    test_metrics = {
        'accuracy': np.array([r['test_metrics']['accuracy'] for r in results]),
        'f1': np.array([r['test_metrics']['f1'] for r in results]),
        'mcc': np.array([r['test_metrics']['mcc'] for r in results])
    }
    
    print(f"\nDataset name: {args.dataset}")
    if args.test_dataset:
        print(f"Test dataset name: {args.test_dataset}")
    else:
        print(f"Test dataset name: {args.dataset}")
    print(f"\nFeature extraction method: {args.method}")    
    print(f"Model name: {args.model}")


    print("\nTest Results across all seeds:")
    for metric, values in test_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.upper()}: {mean:.3f} ± {std:.3f}")