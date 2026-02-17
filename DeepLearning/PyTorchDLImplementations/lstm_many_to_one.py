#!/usr/bin/env python3
"""Bidirectional LSTM many-to-one (sequence -> next token prediction).

Uses synthetic data so you can run it as-is.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

VOCAB = 12
SEQ_LEN = 5
BATCH = 64
EPOCHS = 20
LR = 1e-2
EMBED = 32
HIDDEN = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NextTokenDataset(Dataset):
    """Sequence -> next-token dataset."""

    def __init__(self, n_samples=2000, vocab=VOCAB, seq_len=SEQ_LEN):
        self.x = []
        self.y = []
        for _ in range(n_samples):
            start = random.randint(0, vocab - 1)
            seq = [(start + i) % vocab for i in range(seq_len + 1)]
            self.x.append(seq[:seq_len])
            self.y.append(seq[seq_len])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class NextTokenBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def main():
    set_seed(SEED)

    ds = NextTokenDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = NextTokenBiLSTM(VOCAB, EMBED, HIDDEN).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "bilstm_next_token.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        avg = total / max(count, 1)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch:02d} | loss: {avg:.4f}")

    print(f"Best loss: {best:.4f}")
    print(f"Saved best model to: {best_path}")

    # Example prediction
    model.eval()
    with torch.no_grad():
        sample = torch.tensor([[3, 4, 5, 6, 7]]).to(DEVICE)
        logits = model(sample)
        pred = torch.argmax(logits, dim=1).item()
    print("Input sequence:", sample.cpu().tolist()[0])
    print("Predicted next token:", pred)


if __name__ == "__main__":
    main()
