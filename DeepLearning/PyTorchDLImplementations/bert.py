#!/usr/bin/env python3
"""Minimal BERT-style model with pretraining + example downstream use.

Includes:
- BERT encoder (token + position + segment embeddings)
- Pretraining: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- Fine-tuning: sequence classification (toy)
- Inference: masked token fill + NSP + classification

Synthetic data so it runs end-to-end.
"""

import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -------------------------
# CONFIG
# -------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

MAX_LEN = 16
EMBED = 64
HEADS = 4
FFN_DIM = 128
LAYERS = 2
DROPOUT = 0.1

PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 5
BATCH = 64
LR = 3e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# VOCAB / TOKENIZER
# -------------------------
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
MASK_ID = STOI["[MASK]"]


def tokenize(text: str) -> List[int]:
    return [STOI.get(w, UNK_ID) for w in text.split()]


# -------------------------
# SYNTHETIC CORPUS
# -------------------------
CORPUS = [
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


def make_sentence_pairs(n_samples=2000) -> List[Tuple[str, str, int]]:
    """Return (sent_a, sent_b, label) where label=1 if next sentence, else 0."""
    pairs = []
    for _ in range(n_samples):
        if random.random() < 0.5:
            i = random.randint(0, len(CORPUS) - 2)
            pairs.append((CORPUS[i], CORPUS[i + 1], 1))
        else:
            a = random.choice(CORPUS)
            b = random.choice(CORPUS)
            pairs.append((a, b, 0))
    return pairs


# -------------------------
# DATASET (MLM + NSP)
# -------------------------
class BertPretrainDataset(Dataset):
    def __init__(self, pairs, max_len=MAX_LEN, mlm_prob=0.15):
        self.pairs = pairs
        self.max_len = max_len
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sent_a, sent_b, nsp_label = self.pairs[idx]
        ids_a = tokenize(sent_a)
        ids_b = tokenize(sent_b)

        # [CLS] A [SEP] B [SEP]
        input_ids = [CLS_ID] + ids_a + [SEP_ID] + ids_b + [SEP_ID]
        token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

        # Truncate/pad
        input_ids = input_ids[: self.max_len]
        token_type_ids = token_type_ids[: self.max_len]
        attn_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_len:
            input_ids.append(PAD_ID)
            token_type_ids.append(0)
            attn_mask.append(0)

        # MLM labels
        labels = [-100] * self.max_len
        for i in range(1, self.max_len - 1):
            if input_ids[i] in (PAD_ID, CLS_ID, SEP_ID):
                continue
            if random.random() < self.mlm_prob:
                labels[i] = input_ids[i]
                r = random.random()
                if r < 0.8:
                    input_ids[i] = MASK_ID
                elif r < 0.9:
                    input_ids[i] = random.randint(0, len(VOCAB) - 1)
                else:
                    pass

        return (
            torch.tensor(input_ids),
            torch.tensor(token_type_ids),
            torch.tensor(attn_mask),
            torch.tensor(labels),
            torch.tensor(nsp_label),
        )


# -------------------------
# MODEL
# -------------------------
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
        # attn_mask: [batch, seq] -> [batch, 1, 1, seq]
        mask = attn_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertForPretraining(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.encoder = BertEncoder(vocab, d_model, heads, d_ff, layers, dropout, max_len)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, vocab)
        )
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        cls = enc[:, 0, :]
        mlm_logits = self.mlm_head(enc)
        nsp_logits = self.nsp_head(cls)
        return mlm_logits, nsp_logits


class BertForSequenceClassification(nn.Module):
    def __init__(self, encoder, num_labels=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(EMBED, num_labels)

    def forward(self, input_ids, token_type_ids, attn_mask):
        enc = self.encoder(input_ids, token_type_ids, attn_mask)
        cls = enc[:, 0, :]
        return self.classifier(cls)


# -------------------------
# TRAINING / INFERENCE
# -------------------------

def pretrain():
    pairs = make_sentence_pairs()
    ds = BertPretrainDataset(pairs)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = BertForPretraining(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
    model.to(DEVICE)

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "bert_pretrain.pt")

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for input_ids, token_type_ids, attn_mask, mlm_labels, nsp_labels in dl:
            input_ids = input_ids.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            mlm_labels = mlm_labels.to(DEVICE)
            nsp_labels = nsp_labels.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            mlm_logits, nsp_logits = model(input_ids, token_type_ids, attn_mask)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, len(VOCAB)), mlm_labels.view(-1))
            nsp_loss = nsp_loss_fn(nsp_logits, nsp_labels)
            loss = mlm_loss + nsp_loss
            loss.backward()
            opt.step()

            total += loss.item() * input_ids.size(0)
            count += input_ids.size(0)

        avg = total / max(count, 1)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), best_path)
        print(f"Pretrain Epoch {epoch:02d} | loss: {avg:.4f}")

    print(f"Best pretrain loss: {best:.4f}")
    print(f"Saved pretrain model to: {best_path}")
    return best_path


def finetune_sequence_classification(pretrain_path):
    # Simple synthetic labels: 1 if sentence contains 'king' or 'queen', else 0
    sentences = CORPUS * 20
    labels = [1 if ("king" in s or "queen" in s) else 0 for s in sentences]

    # Build inputs
    inputs = []
    for s in sentences:
        ids = [CLS_ID] + tokenize(s) + [SEP_ID]
        ids = ids[:MAX_LEN]
        attn = [1] * len(ids)
        seg = [0] * len(ids)
        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            attn.append(0)
            seg.append(0)
        inputs.append((ids, seg, attn))

    # Dataset
    class ClsDS(Dataset):
        def __len__(self):
            return len(inputs)

        def __getitem__(self, idx):
            ids, seg, attn = inputs[idx]
            return (
                torch.tensor(ids),
                torch.tensor(seg),
                torch.tensor(attn),
                torch.tensor(labels[idx]),
            )

    ds = ClsDS()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
    model = BertForSequenceClassification(encoder, num_labels=2).to(DEVICE)

    # Load pretrained encoder weights
    state = torch.load(pretrain_path, map_location=DEVICE)
    model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")})

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    best_path = os.path.join(OUT_DIR, "bert_finetune.pt")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for input_ids, token_type_ids, attn_mask, y in dl:
            input_ids = input_ids.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids, token_type_ids, attn_mask)
            loss = loss_fn(logits, y)
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
    print(f"Saved finetune model to: {best_path}")
    return best_path


def inference(pretrain_path, finetune_path):
    # Load pretrain model for MLM/NSP
    pre = BertForPretraining(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN).to(DEVICE)
    pre.load_state_dict(torch.load(pretrain_path, map_location=DEVICE))
    pre.eval()

    # Masked token prediction
    sentence = "paris is [MASK] france"
    ids = [CLS_ID] + [STOI.get(w, UNK_ID) for w in sentence.split()] + [SEP_ID]
    ids = ids[:MAX_LEN]
    seg = [0] * len(ids)
    attn = [1] * len(ids)
    while len(ids) < MAX_LEN:
        ids.append(PAD_ID)
        seg.append(0)
        attn.append(0)

    input_ids = torch.tensor([ids]).to(DEVICE)
    token_type_ids = torch.tensor([seg]).to(DEVICE)
    attn_mask = torch.tensor([attn]).to(DEVICE)

    with torch.no_grad():
        mlm_logits, nsp_logits = pre(input_ids, token_type_ids, attn_mask)
        mask_idx = ids.index(MASK_ID)
        pred_id = torch.argmax(mlm_logits[0, mask_idx]).item()
        print("Masked sentence:", sentence)
        print("Predicted token:", ITOS[pred_id])

    # NSP prediction
    sent_a = "paris is a city"
    sent_b = "france is a country"
    ids_a = tokenize(sent_a)
    ids_b = tokenize(sent_b)
    ids = [CLS_ID] + ids_a + [SEP_ID] + ids_b + [SEP_ID]
    seg = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)
    ids = ids[:MAX_LEN]
    seg = seg[:MAX_LEN]
    attn = [1] * len(ids)
    while len(ids) < MAX_LEN:
        ids.append(PAD_ID)
        seg.append(0)
        attn.append(0)

    input_ids = torch.tensor([ids]).to(DEVICE)
    token_type_ids = torch.tensor([seg]).to(DEVICE)
    attn_mask = torch.tensor([attn]).to(DEVICE)

    with torch.no_grad():
        _, nsp_logits = pre(input_ids, token_type_ids, attn_mask)
        nsp_pred = torch.argmax(nsp_logits, dim=1).item()
        print("NSP pair:", sent_a, "|", sent_b)
        print("NSP prediction (1=next,0=not):", nsp_pred)

    # Sequence classification prediction
    encoder = BertEncoder(len(VOCAB), EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, MAX_LEN)
    clf = BertForSequenceClassification(encoder, num_labels=2).to(DEVICE)
    clf.load_state_dict(torch.load(finetune_path, map_location=DEVICE))
    clf.eval()

    test_sent = "queen is a woman"
    ids = [CLS_ID] + tokenize(test_sent) + [SEP_ID]
    ids = ids[:MAX_LEN]
    seg = [0] * len(ids)
    attn = [1] * len(ids)
    while len(ids) < MAX_LEN:
        ids.append(PAD_ID)
        seg.append(0)
        attn.append(0)

    input_ids = torch.tensor([ids]).to(DEVICE)
    token_type_ids = torch.tensor([seg]).to(DEVICE)
    attn_mask = torch.tensor([attn]).to(DEVICE)

    with torch.no_grad():
        logits = clf(input_ids, token_type_ids, attn_mask)
        pred = torch.argmax(logits, dim=1).item()
        print("Classification sentence:", test_sent)
        print("Classification prediction (1=royalty,0=other):", pred)


def main():
    set_seed(SEED)
    pretrain_path = pretrain()
    finetune_path = finetune_sequence_classification(pretrain_path)
    inference(pretrain_path, finetune_path)


if __name__ == "__main__":
    main()
