#!/usr/bin/env python3
"""Transformer inference using finetuned weights.

Loads outputs/transformer_finetune.pt and runs greedy decoding.
"""

import math
import os
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs"

VOCAB = 20
SEQ_LEN = 8
EMBED = 64
HEADS = 4
FFN_DIM = 128
LAYERS = 2
DROPOUT = 0.1


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

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        q = self.wq(q).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        k = self.wk(k).view(b, -1, self.heads, self.d_head).transpose(1, 2)
        v = self.wv(v).view(b, -1, self.heads, self.d_head).transpose(1, 2)
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

    def forward(self, x, src_mask=None):
        attn = self.attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop(attn))
        ffn = self.ffn(x)
        x = self.norm2(x + self.drop(ffn))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(attn))
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop(attn2))
        ffn = self.ffn(x)
        x = self.norm3(x + self.drop(ffn))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab, d_model, heads, d_ff, layers, dropout=0.0, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab)

    def encode(self, src, src_mask=None):
        x = self.drop(self.pos(self.tok(src)))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.drop(self.pos(self.tok(tgt)))
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.fc(x)

    def forward(self, src, tgt, tgt_mask=None, src_mask=None):
        enc_out = self.encode(src, src_mask)
        logits = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return logits


def subsequent_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)


def main():
    model = Transformer(VOCAB, EMBED, HEADS, FFN_DIM, LAYERS, DROPOUT, max_len=SEQ_LEN)
    model.to(DEVICE)

    ckpt = os.path.join(OUT_DIR, "transformer_finetune.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Run transformer_finetune.py first.")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).to(DEVICE)
        tgt = src[:, :1]
        for _ in range(SEQ_LEN):
            tgt_mask = subsequent_mask(tgt.size(1)).to(DEVICE)
            logits = model(src, tgt, tgt_mask=tgt_mask)
            next_tok = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
        print("Input sequence:", src.cpu().tolist()[0])
        print("Predicted sequence:", tgt[:, 1:].cpu().tolist()[0])


if __name__ == "__main__":
    main()
