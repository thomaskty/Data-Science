#!/usr/bin/env python3
"""Minimal GPT-style decoder-only Transformer.

Synthetic task: next-token prediction on simple sequences.
Includes:
- Token + positional embeddings
- Causal self-attention
- Decoder blocks
- Training + saving best model
- Inference: generate new tokens from a prompt
"""

import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

VOCAB = 30
SEQ_LEN = 12
BATCH = 64
EPOCHS = 30
LR = 3e-4
EMBED = 64
HEADS = 4
FFN_DIM = 128
LAYERS = 2
DROPOUT = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NextTokenDataset(Dataset):
    """Synthetic sequences: increasing modulo vocab."""

    def __init__(self, n_samples=4000, vocab=VOCAB, seq_len=SEQ_LEN):
        self.x = []
        self.y = []
        for _ in range(n_samples):
            start = random.randint(0, vocab - 1)
            seq = [(start + i) % vocab for i in range(seq_len + 1)]
            self.x.append(seq[:seq_len])
            self.y.append(seq[1:])  # next-token targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b = x.size(0)
        q = self.wq(x).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, -1, self.heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.wo(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.attn(x, mask)
        x = self.norm1(x + self.drop(attn))
        ffn = self.ffn(x)
        x = self.norm2(x + self.drop(ffn))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x, mask=None):
        x = self.drop(self.pos(self.tok(x)))
        for block in self.blocks:
            x = block(x, mask)
        return self.fc(x)


def subsequent_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)


def generate(model, prompt, max_new_tokens=10):
    model.eval()
    tokens = prompt[:]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens]).to(DEVICE)
            mask = subsequent_mask(x.size(1)).to(DEVICE)
            logits = model(x, mask)
            next_tok = torch.argmax(logits[:, -1, :], dim=1).item()
            tokens.append(next_tok)
    return tokens


def main():
    set_seed(SEED)

    ds = NextTokenDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = GPT(VOCAB, EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, max_len=SEQ_LEN)
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "gpt_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            mask = subsequent_mask(xb.size(1)).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mask)
            loss = loss_fn(logits.view(-1, VOCAB), yb.view(-1))
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

    # Inference: reload and generate from a prompt
    infer_model = GPT(VOCAB, EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, max_len=SEQ_LEN)
    infer_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    infer_model.to(DEVICE)

    prompt = [5, 6, 7]
    generated = generate(infer_model, prompt, max_new_tokens=10)
    print("Prompt:", prompt)
    print("Generated:", generated)


if __name__ == "__main__":
    main()
