import json
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from configs import get_model, get_task
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier on extracted features.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., vul).")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., codebert).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.000001, help="Learning rate for the optimizer.")
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

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_acc = 0
        train_loss = 0
        all_preds = []
        all_labels = []

        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            train_loss += loss.item()

            preds = outputs.argmax(dim=1)
            train_acc += (preds == y).float().mean()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        if epoch % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1 Score: {train_f1:.4f}"
            )

            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for x, y in val_loader:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    val_accuracy += (preds == y).float().mean().item()

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            print(
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
                f"Validation F1 Score: {val_f1:.4f}"
            )

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            predicted = outputs.argmax(dim=1)

            # Accuracy computation
            correct += (predicted == y).sum().item()
            total += y.size(0)

            # Collect predictions and labels for F1
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_accuracy = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    args = parse_args()

    # Load task and model configurations
    task = get_task(args.task)
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

    # Initialize and train the model
    model = Classifier(input_size=feat_dim, num_classes=len(cat2id))
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)

    # Test the model
    test_model(model, test_loader)