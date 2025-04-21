import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from configs import get_model, get_task
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import wandb
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier on extracted features.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., vul).")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., codebert).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., devign, primevul).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--method", type=str, default="CLS", choices=["CLS", "AVG", "MAX"], help="Method to use for feature extraction.")

    return parser.parse_args()

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
            # Extract features based on the specified method
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

def get_file_path(task_config, model_config, split):
    base_dir = task_config["base_dir"]
    model_name = model_config["model_path"].split("/")[-1]
    return f"{base_dir}/features/{model_name}/{split}_{model_name}_features.jsonl"

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out

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
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
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
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
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
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            predicted = outputs.argmax(dim=1)

            # Accuracy computation
            total += y.size(0)

            # Collect predictions and labels for F1
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
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

    # Load task and model configurations
    task = get_task(args.task, args.dataset)
    model_config = get_model(args.model)

    # Get hyperparameters from command line arguments
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    method = args.method

    # Get the file paths for train, valid, and test datasets
    train_file = get_file_path(task, model_config, "train")
    valid_file = get_file_path(task, model_config, "valid")
    test_file = get_file_path(task, model_config, "test")

    # Load training data
    train_X, train_y, feat_dim, cat2id, id2cat = load_data(train_file, method)

    # Load validation data
    valid_X, valid_y, _, _, _ = load_data(valid_file, method)

    # Load testing data
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

    # Create DataLoader objects
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    num_runs = 10
    best_val_f1 = 0.0
    best_model_path = None
    best_checkpoint = None

    for run in range(num_runs):
        run_name = f"{args.task}_{args.dataset}_{args.model}_{args.method}_run{run+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="code-llm-embedding-eval",
            name=run_name,
            config={
                "task": args.task,
                "dataset": args.dataset,
                "model": args.model,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "method": method,
                "run": run + 1
            }
        )

        model = Classifier(input_size=feat_dim, num_classes=len(cat2id))
        model_path = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

        checkpoint = torch.load(model_path, weights_only=False)
        val_f1 = checkpoint['val_f1']

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = f"checkpoints/best_overall_{args.task}_{args.dataset}_{args.model}_{args.method}.pt"
            best_checkpoint = checkpoint
            torch.save(checkpoint, best_model_path)
            print(f"New best overall model found at run {run+1} with val F1: {val_f1:.4f}")

        wandb.finish()

    print(f"\nBest model across all runs: {best_model_path} (val F1: {best_val_f1:.4f})")
    model = Classifier(input_size=feat_dim, num_classes=len(cat2id))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_model(model, test_loader)