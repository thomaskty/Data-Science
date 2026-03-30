#!/usr/bin/env python3
"""Text/sequence autoencoder (RNN) on synthetic token sequences.

- Encoder GRU -> latent (hidden state)
- Decoder GRU -> reconstruct input tokens
- Teacher forcing during training
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = 20
SEQ_LEN = 8
BATCH = 64
EPOCHS = 20
LR = 1e-3
EMBED = 32
HIDDEN = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, n_samples=3000, vocab=VOCAB, seq_len=SEQ_LEN):
        self.x = []
        for _ in range(n_samples):
            start = random.randint(0, vocab - 1)
            seq = [(start + i) % vocab for i in range(seq_len)]
            self.x.append(seq)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        return x


class Encoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.rnn(emb)
        return h  # [1, batch, hidden]


class Decoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, h):
        emb = self.emb(x)
        out, _ = self.rnn(emb, h)
        logits = self.fc(out)
        return logits


class Autoencoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.encoder = Encoder(vocab, embed, hidden)
        self.decoder = Decoder(vocab, embed, hidden)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.decoder(x, h)  # teacher forcing
        return logits


def main():
    set_seed(SEED)

    ds = SeqDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = Autoencoder(VOCAB, EMBED, HIDDEN).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, VOCAB), xb.view(-1))
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        avg = total / max(count, 1)
        print(f"Epoch {epoch:02d} | loss: {avg:.4f}")

    # Inference example
    model.eval()
    with torch.no_grad():
        sample = ds[0].unsqueeze(0).to(DEVICE)
        logits = model(sample)
        recon = torch.argmax(logits, dim=-1).cpu().tolist()[0]
    print("Input:", sample.cpu().tolist()[0])
    print("Recon:", recon)


if __name__ == "__main__":
    main()
