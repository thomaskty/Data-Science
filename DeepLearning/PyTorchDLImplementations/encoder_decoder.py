#!/usr/bin/env python3
"""Encoder-Decoder (seq2seq) without attention.

Synthetic task: token-wise shift by +1 (mod vocab).
Shows:
- Encoder/Decoder with GRU
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
        _, hidden = self.rnn(emb)
        return hidden  # [1, batch, hidden]


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x: [batch, 1]
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out[:, -1, :])
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing=0.5):
        # src: [batch, seq_len], tgt: [batch, seq_len]
        batch, seq_len = tgt.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch, seq_len, vocab_size, device=src.device)
        hidden = self.encoder(src)

        # Start token = first target token (teacher forcing style)
        input_tok = tgt[:, 0].unsqueeze(1)
        for t in range(seq_len):
            logits, hidden = self.decoder(input_tok, hidden)
            outputs[:, t, :] = logits
            use_teacher = random.random() < teacher_forcing
            next_tok = tgt[:, t] if use_teacher else torch.argmax(logits, dim=1)
            input_tok = next_tok.unsqueeze(1)

        return outputs

    def greedy_decode(self, src, max_len):
        self.eval()
        with torch.no_grad():
            hidden = self.encoder(src)
            # start from first src token
            input_tok = src[:, 0].unsqueeze(1)
            outputs = []
            for _ in range(max_len):
                logits, hidden = self.decoder(input_tok, hidden)
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
    model = Seq2Seq(enc, dec).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "seq2seq_best.pt")

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
