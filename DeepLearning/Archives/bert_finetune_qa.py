#!/usr/bin/env python3
"""BERT fine-tuning for extractive QA (start/end positions).

Loads outputs/bert_pretrain.pt
Saves finetuned model to outputs/bert_finetune_qa.pt
"""

import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

MAX_LEN = 16
EMBED = 64
HEADS = 4
FFN_DIM = 128
LAYERS = 2
DROPOUT = 0.1

EPOCHS = 5
BATCH = 32
LR = 3e-4

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = [
    "king",
    "queen",
    "man",
    "woman",
    "boy",
    "girl",
    "paris",
    "london",
    "france",
    "england",
    "city",
    "country",
    "is",
    "a",
    "in",
    "and",
    "are",
    "royalty",
    "children",
]
VOCAB = SPECIAL + WORDS
STOI = {w: i for i, w in enumerate(VOCAB)}
ITOS = {i: w for w, i in STOI.items()}

PAD_ID = STOI["[PAD]"]
UNK_ID = STOI["[UNK]"]
CLS_ID = STOI["[CLS]"]
SEP_ID = STOI["[SEP]"]


CONTEXTS = [
    "paris is a city in france",
    "london is a city in england",
    "king and queen are royalty",
]
QUESTIONS = [
    ("what is the city", "paris"),
    ("what is the city", "london"),
    ("who are royalty", "king"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[int]:
    return [STOI.get(w, UNK_ID) for w in text.split()]


class QADataset(Dataset):
    def __init__(self, n_samples=500):
        self.samples = []
        for _ in range(n_samples):
            ctx = random.choice(CONTEXTS)
            q, ans = random.choice(QUESTIONS)
            self.samples.append((q, ctx, ans))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, ctx, ans = self.samples[idx]
        q_ids = tokenize(q)
        ctx_ids = tokenize(ctx)

        # [CLS] Q [SEP] C [SEP]
        input_ids = [CLS_ID] + q_ids + [SEP_ID] + ctx_ids + [SEP_ID]
        token_type_ids = [0] * (len(q_ids) + 2) + [1] * (len(ctx_ids) + 1)

        # find answer position in context
        ans_id = STOI.get(ans, UNK_ID)
        ctx_start = 1 + len(q_ids) + 1  # CLS + Q + SEP
        try:
            ans_pos = ctx_ids.index(ans_id)
            start = ctx_start + ans_pos
            end = start
        except ValueError:
            start = 0
            end = 0

        input_ids = input_ids[:MAX_LEN]
        token_type_ids = token_type_ids[:MAX_LEN]
        attn = [1] * len(input_ids)

        while len(input_ids) < MAX_LEN:
            input_ids.append(PAD_ID)
            token_type_ids.append(0)
            attn.append(0)

        return (
            torch.tensor(input_ids),
            torch.tensor(token_type_ids),
            torch.tensor(attn),
            torch.tensor(start),
            torch.tensor(end),
        )


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


class EncoderLayer(nn.Module):
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


class BertEncoder(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.seg = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )

    def forward(self, input_ids, token_type_ids, attn_mask):
        x = self.tok(input_ids) + self.seg(token_type_ids)
        x = self.drop(self.pos(x))
        mask = attn_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertForQA(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.qa_head = nn.Linear(EMBED, 2)  # start/end

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        logits = self.qa_head(enc)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


def main():
    set_seed(SEED)

    ds = QADataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
    model = BertForQA(encoder).to(DEVICE)

    pre_path = os.path.join(OUT_DIR, "bert_pretrain.pt")
    if os.path.exists(pre_path):
        state = torch.load(pre_path, map_location=DEVICE)
        model.encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        )

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    best_path = os.path.join(OUT_DIR, "bert_finetune_qa.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for input_ids, token_type_ids, attn, start_pos, end_pos in dl:
            input_ids = input_ids.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            attn = attn.to(DEVICE)
            start_pos = start_pos.to(DEVICE)
            end_pos = end_pos.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            start_logits, end_logits = model(input_ids, token_type_ids, attn)
            loss = loss_fn(start_logits, start_pos) + loss_fn(end_logits, end_pos)
            loss.backward()
            opt.step()

            total += loss.item() * input_ids.size(0)
            count += input_ids.size(0)

        avg = total / max(count, 1)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), best_path)
        print(f"Finetune Epoch {epoch:02d} | loss: {avg:.4f}")

    print(f"Best finetune loss: {best:.4f}")
    print(f"Saved finetuned model to: {best_path}")


if __name__ == "__main__":
    main()
