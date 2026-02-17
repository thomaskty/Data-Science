#!/usr/bin/env python3
"""BERT fine-tuning for token classification (NER-style).

Loads outputs/bert_pretrain.pt
Saves finetuned model to outputs/bert_finetune_token.pt
"""

import math
import os
import random
from typing import List

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
BATCH = 64
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

PAD_ID = STOI["[PAD]"]
UNK_ID = STOI["[UNK]"]
CLS_ID = STOI["[CLS]"]
SEP_ID = STOI["[SEP]"]

ENTITY_TOKENS = {"paris", "london", "france", "england"}  # label 1 for these

CORPUS = [
    "paris is a city",
    "london is a city",
    "france is a country",
    "england is a country",
    "paris is in france",
    "london is in england",
    "king is a man",
    "queen is a woman",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[int]:
    return [STOI.get(w, UNK_ID) for w in text.split()]


class TokenDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx].split()
        labels = [1 if w in ENTITY_TOKENS else 0 for w in words]

        ids = [CLS_ID] + tokenize(self.sentences[idx]) + [SEP_ID]
        ids = ids[:MAX_LEN]
        attn = [1] * len(ids)
        seg = [0] * len(ids)

        # labels align with tokens (CLS/SEP get -100 to ignore)
        token_labels = [-100] + labels + [-100]
        token_labels = token_labels[:MAX_LEN]

        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            attn.append(0)
            seg.append(0)
            token_labels.append(-100)

        return (
            torch.tensor(ids),
            torch.tensor(seg),
            torch.tensor(attn),
            torch.tensor(token_labels),
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


class BertForTokenClassification(nn.Module):
    def __init__(self, encoder, num_labels=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(EMBED, num_labels)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        return self.classifier(enc)


def main():
    set_seed(SEED)

    sentences = CORPUS * 30
    ds = TokenDataset(sentences)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
    model = BertForTokenClassification(encoder, num_labels=2).to(DEVICE)

    pre_path = os.path.join(OUT_DIR, "bert_pretrain.pt")
    if os.path.exists(pre_path):
        state = torch.load(pre_path, map_location=DEVICE)
        model.encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    best_path = os.path.join(OUT_DIR, "bert_finetune_token.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for input_ids, token_type_ids, attn_mask, labels in dl:
            input_ids = input_ids.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits = model(input_ids, token_type_ids, attn_mask)
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
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
