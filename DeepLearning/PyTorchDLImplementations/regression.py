#!/usr/bin/env python3
"""Simple PyTorch regression example for tabular numerical data.

What this shows:
- NeuralNetwork class with __init__, forward, train, predict, evaluate
- Synthetic dataset + DataLoaders (mini-batch training)
- Train/test loss logging per epoch
- Save best model by test loss
- Predict on a single sample and on a batch
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
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs, test_loader=None, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        best_test = float("inf")
        best_path = os.path.join(out_dir, "best_model.pt")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_sum = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()

                train_sum += loss.item() * xb.size(0)
                train_count += xb.size(0)

            train_loss = train_sum / max(train_count, 1)

            if test_loader is not None:
                test_loss = self.evaluate(test_loader)
                if test_loss < best_test:
                    best_test = test_loss
                    torch.save(self.model.state_dict(), best_path)
                print(
                    f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | test MSE: {test_loss:.6f}"
                )
            else:
                print(f"Epoch {epoch:03d} | train MSE: {train_loss:.6f}")

        if test_loader is not None:
            print(f"Best test MSE: {best_test:.6f}")
            print(f"Saved best model to: {best_path}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def evaluate(self, data_loader):
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                total += loss.item() * xb.size(0)
                count += xb.size(0)
        return total / max(count, 1)


# -------------------------
# DATA (synthetic example)
# -------------------------
def make_synthetic(n_samples=1000, input_dim=5, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, input_dim)).astype(np.float32)
    true_w = rng.normal(0, 1, size=(input_dim, 1)).astype(np.float32)
    y = X @ true_w + noise * rng.normal(0, 1, size=(n_samples, 1)).astype(np.float32)
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
    X, y = make_synthetic(n_samples=2000, input_dim=input_dim, noise=0.2)

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
    pred_single = trainer.predict(sample)
    print("Single prediction:", pred_single.cpu().numpy())

    # Batch prediction
    batch = torch.from_numpy(X_test[:10])
    pred_batch = trainer.predict(batch)
    print("Batch prediction shape:", pred_batch.shape)


if __name__ == "__main__":
    main()
