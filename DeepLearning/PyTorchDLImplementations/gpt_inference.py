#!/usr/bin/env python3
"""GPT inference examples.

- Generation using pretrained LM
- Classification using finetuned classifier
"""

import math
import os
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

VOCAB = 30
SEQ_LEN = 12
EMBED = 64
HEADS = 4
FFN_DIM = 128
LAYERS = 2
DROPOUT = 0.1
NUM_LABELS = 2


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


class GPTBackbone(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.drop(self.pos(self.tok(x)))
        for block in self.blocks:
            x = block(x, mask)
        return x


class GPTForClassification(nn.Module):
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(EMBED, num_labels)

    def forward(self, x, mask=None):
        h = self.backbone(x, mask)
        last = h[:, -1, :]
        return self.classifier(last)


def subsequent_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)


def generate(model, prompt, max_new_tokens=10):
    model.eval()
    tokens = prompt[:]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # keep context length <= SEQ_LEN
            ctx = tokens[-SEQ_LEN:]
            x = torch.tensor([ctx]).to(DEVICE)
            mask = subsequent_mask(x.size(1)).to(DEVICE)
            logits = model(x, mask)
            next_tok = torch.argmax(logits[:, -1, :], dim=1).item()
            tokens.append(next_tok)
    return tokens


def main():
    # Generation using pretrained GPT
    pre_ckpt = os.path.join(OUT_DIR, "gpt_pretrain.pt")
    if os.path.exists(pre_ckpt):
        gpt = GPT(VOCAB, EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, max_len=SEQ_LEN).to(DEVICE)
        gpt.load_state_dict(torch.load(pre_ckpt, map_location=DEVICE))
        prompt = [5, 6, 7]
        generated = generate(gpt, prompt, max_new_tokens=10)
        print("Prompt:", prompt)
        print("Generated:", generated)
    else:
        print("Pretrained GPT not found. Run gpt_pretrain.py first.")

    # Classification using finetuned GPT
    fin_ckpt = os.path.join(OUT_DIR, "gpt_finetune_cls.pt")
    if os.path.exists(fin_ckpt):
        backbone = GPTBackbone(VOCAB, EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, max_len=SEQ_LEN)
        clf = GPTForClassification(backbone, num_labels=NUM_LABELS).to(DEVICE)
        clf.load_state_dict(torch.load(fin_ckpt, map_location=DEVICE))
        clf.eval()
        sample = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]]).to(DEVICE)
        mask = subsequent_mask(sample.size(1)).to(DEVICE)
        with torch.no_grad():
            logits = clf(sample, mask)
            pred = torch.argmax(logits, dim=1).item()
        print("Classification sample:", sample.cpu().tolist()[0])
        print("Predicted label:", pred)
    else:
        print("Finetuned GPT classifier not found. Run gpt_finetune.py first.")


if __name__ == "__main__":
    main()
