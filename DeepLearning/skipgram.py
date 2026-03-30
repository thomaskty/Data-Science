#!/usr/bin/env python3
"""Skip-gram training on synthetic text.

Industry-style basics:
- Build vocab + dataset
- Train loop with mini-batches
- Save best model by validation loss
- Example predictions
"""

import os
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# -------------------------
# CONFIG
# -------------------------
SEED = 42
EMBED_DIM = 32
WINDOW = 2
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-2
OUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_synthetic_text():
    sentences = [
        "king is a man",
        "queen is a woman",
        "boy is a man",
        "girl is a woman",
        "paris is a city",
        "london is a city",
        "france is a country",
        "england is a country",
        "paris is in france",
        "london is in england",
        "king and queen are royalty",
        "boy and girl are children",
    ]
    return [s.split() for s in sentences]


def build_vocab(tokens):
    counts = Counter(tokens)
    vocab = ["<unk>"] + sorted(counts.keys())
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


class SkipGramDataset(Dataset):
    def __init__(self, corpus, stoi, window):
        self.samples = []
        for sent in corpus:
            idxs = [stoi.get(w, 0) for w in sent]
            for i in range(len(idxs)):
                center = idxs[i]
                for j in range(max(0, i - window), min(len(idxs), i + window + 1)):
                    if i == j:
                        continue
                    context = idxs[j]
                    self.samples.append((center, context))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center, context = self.samples[idx]
        return torch.tensor(center), torch.tensor(context)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch]
        emb = self.emb(x)  # [batch, embed_dim]
        logits = self.linear(emb)
        return logits


def train_model(model, train_loader, val_loader):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    os.makedirs(OUT_DIR, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "skipgram_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0
        count = 0
        for center, ctx in train_loader:
            center = center.to(DEVICE)
            ctx = ctx.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(center)
            loss = loss_fn(logits, ctx)
            loss.backward()
            opt.step()
            train_sum += loss.item() * center.size(0)
            count += center.size(0)
        train_loss = train_sum / max(count, 1)

        val_loss = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

    print(f"Best val loss: {best_val:.4f}")
    print(f"Saved best model to: {best_path}")


def evaluate(model, data_loader, loss_fn):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for center, ctx in data_loader:
            center = center.to(DEVICE)
            ctx = ctx.to(DEVICE)
            logits = model(center)
            loss = loss_fn(logits, ctx)
            total += loss.item() * center.size(0)
            count += center.size(0)
    return total / max(count, 1)


def predict(model, stoi, itos, center_word):
    model.eval()
    idx = stoi.get(center_word, 0)
    x = torch.tensor([idx]).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        topk = torch.topk(probs, k=5)
    return [(itos[i.item()], p.item()) for i, p in zip(topk.indices, topk.values)]


def main():
    set_seed(SEED)

    corpus = make_synthetic_text()
    tokens = [w for sent in corpus for w in sent]
    stoi, itos = build_vocab(tokens)

    dataset = SkipGramDataset(corpus, stoi, WINDOW)

    # Split train/val
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(n * 0.8)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False
    )

    model = SkipGramModel(vocab_size=len(stoi), embed_dim=EMBED_DIM)
    model.to(DEVICE)

    train_model(model, train_loader, val_loader)

    # Example predictions
    print("\nExample predictions:")
    print("Center word: 'paris'")
    preds = predict(model, stoi, itos, "paris")
    print(preds)


if __name__ == "__main__":
    main()
