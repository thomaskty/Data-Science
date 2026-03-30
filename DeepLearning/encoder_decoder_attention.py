#!/usr/bin/env python3
"""Encoder-Decoder with dot-product attention.

Synthetic task: token-wise shift by +1 (mod vocab).
Shows:
- Encoder outputs all hidden states
- Decoder attends to encoder outputs each step
- Teacher forcing during training
- Greedy decoding for inference
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
SEQ_LEN = 6
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


class Seq2SeqDataset(Dataset):
    def __init__(self, n_samples=2000, vocab=VOCAB, seq_len=SEQ_LEN):
        self.x = []
        self.y = []
        for _ in range(n_samples):
            seq = [random.randint(0, vocab - 1) for _ in range(seq_len)]
            tgt = [((t + 1) % vocab) for t in seq]
            self.x.append(seq)
            self.y.append(tgt)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden  # outputs: [batch, seq_len, hidden]


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, query, keys, values):
        # query: [batch, hidden], keys/values: [batch, seq_len, hidden]
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) / self.scale
        attn = torch.softmax(scores, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.attn = Attention(hidden_dim)

    def forward(self, x, hidden, enc_outputs):
        emb = self.emb(x)  # [batch, 1, embed]
        query = hidden[-1]  # [batch, hidden]
        context, _ = self.attn(query, enc_outputs, enc_outputs)
        context = context.unsqueeze(1)  # [batch, 1, hidden]
        rnn_in = torch.cat([emb, context], dim=2)
        out, hidden = self.rnn(rnn_in, hidden)
        out = out[:, -1, :]
        logits = self.fc(torch.cat([out, context.squeeze(1)], dim=1))
        return logits, hidden


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing=0.5):
        batch, seq_len = tgt.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch, seq_len, vocab_size, device=src.device)
        enc_outputs, hidden = self.encoder(src)

        input_tok = tgt[:, 0].unsqueeze(1)
        for t in range(seq_len):
            logits, hidden = self.decoder(input_tok, hidden, enc_outputs)
            outputs[:, t, :] = logits
            use_teacher = random.random() < teacher_forcing
            next_tok = tgt[:, t] if use_teacher else torch.argmax(logits, dim=1)
            input_tok = next_tok.unsqueeze(1)

        return outputs

    def greedy_decode(self, src, max_len):
        self.eval()
        with torch.no_grad():
            enc_outputs, hidden = self.encoder(src)
            input_tok = src[:, 0].unsqueeze(1)
            outputs = []
            for _ in range(max_len):
                logits, hidden = self.decoder(input_tok, hidden, enc_outputs)
                pred = torch.argmax(logits, dim=1)
                outputs.append(pred)
                input_tok = pred.unsqueeze(1)
            return torch.stack(outputs, dim=1)


def main():
    set_seed(SEED)

    ds = Seq2SeqDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    enc = Encoder(VOCAB, EMBED, HIDDEN)
    dec = Decoder(VOCAB, EMBED, HIDDEN)
    model = Seq2SeqAttn(enc, dec).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "seq2seq_attn_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, yb, teacher_forcing=0.7)
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

    # Example prediction
    model.eval()
    with torch.no_grad():
        sample = torch.tensor([[1, 2, 3, 4, 5, 6]]).to(DEVICE)
        preds = model.greedy_decode(sample, max_len=SEQ_LEN)
    print("Input sequence:", sample.cpu().tolist()[0])
    print("Predicted sequence:", preds.cpu().tolist()[0])


if __name__ == "__main__":
    main()
