#!/usr/bin/env python3
"""Simple PyTorch binary classification example for tabular numerical data.

What this shows:
- NeuralNetwork class with __init__ and forward
- Trainer class with train/evaluate/predict
- Synthetic dataset + DataLoaders (mini-batch training)
- Train/test loss + accuracy per epoch
- Save best model by test loss
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, no_hidden_layers, hidden_dim, output_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(no_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trainer:
    def __init__(self, model, learning_rate, device):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs, test_loader=None, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        best_test = float("inf")
        best_path = os.path.join(out_dir, "best_model.pt")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_sum = 0.0
            train_correct = 0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()

                train_sum += loss.item() * xb.size(0)
                train_count += xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                train_correct += (preds == yb).sum().item()

            train_loss = train_sum / max(train_count, 1)
            train_acc = train_correct / max(train_count, 1)

            if test_loader is not None:
                test_loss, test_acc = self.evaluate(test_loader)
                if test_loss < best_test:
                    best_test = test_loss
                    torch.save(self.model.state_dict(), best_path)
                print(
                    f"Epoch {epoch:03d} | train loss: {train_loss:.6f} | train acc: {train_acc:.4f} "
                    f"| test loss: {test_loss:.6f} | test acc: {test_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | train loss: {train_loss:.6f} | train acc: {train_acc:.4f}"
                )

        if test_loader is not None:
            print(f"Best test loss: {best_test:.6f}")
            print(f"Saved best model to: {best_path}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits)
            return probs

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                total_loss += loss.item() * xb.size(0)
                total_count += xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (preds == yb).sum().item()
        avg_loss = total_loss / max(total_count, 1)
        acc = total_correct / max(total_count, 1)
        return avg_loss, acc


# -------------------------
# DATA (synthetic example)
# -------------------------
def make_synthetic(n_samples=2000, input_dim=5, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, input_dim)).astype(np.float32)
    w = rng.normal(0, 1, size=(input_dim, 1)).astype(np.float32)
    logits = X @ w + noise * rng.normal(0, 1, size=(n_samples, 1)).astype(np.float32)
    y = (logits > 0).astype(np.float32)  # binary labels 0/1
    return X, y


def standardize(x_train, x_test):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def main():
    # Config
    input_dim = 5
    output_dim = 1
    hidden_dim = 64
    no_hidden_layers = 2
    learning_rate = 1e-3
    batch_size = 64
    epochs = 50
    test_split = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create synthetic data
    X, y = make_synthetic(n_samples=2000, input_dim=input_dim, noise=0.5)

    # Train/test split
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1.0 - test_split))
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize using training stats
    X_train, X_test = standardize(X_train, X_test)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build and train model
    net = NeuralNetwork(
        input_dim=input_dim,
        no_hidden_layers=no_hidden_layers,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    trainer = Trainer(net, learning_rate=learning_rate, device=device)
    trainer.train(
        train_loader, epochs=epochs, test_loader=test_loader, out_dir="outputs"
    )

    # -------------------------
    # Production-style prediction
    # -------------------------
    # Single sample prediction
    sample = torch.from_numpy(X_test[:1])
    prob_single = trainer.predict(sample)
    print("Single prediction (probability):", prob_single.cpu().numpy())

    # Batch prediction
    batch = torch.from_numpy(X_test[:10])
    prob_batch = trainer.predict(batch)
    print("Batch prediction shape:", prob_batch.shape)


if __name__ == "__main__":
    main()
