#!/usr/bin/env python3
"""Text/sequence VAE (RNN) on synthetic token sequences.

- Encoder GRU -> mean/logvar
- Reparameterization -> latent z
- Decoder GRU -> reconstruct input tokens
- Loss = reconstruction + KL
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
LATENT = 32


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
        return torch.tensor(self.x[idx])


class Encoder(nn.Module):
    def __init__(self, vocab, embed, hidden, latent):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.rnn(emb)
        h = h.squeeze(0)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, vocab, embed, hidden, latent):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.fc = nn.Linear(latent, hidden)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, z):
        h0 = torch.tanh(self.fc(z)).unsqueeze(0)
        emb = self.emb(x)
        out, _ = self.rnn(emb, h0)
        logits = self.out(out)
        return logits


class TextVAE(nn.Module):
    def __init__(self, vocab, embed, hidden, latent):
        super().__init__()
        self.encoder = Encoder(vocab, embed, hidden, latent)
        self.decoder = Decoder(vocab, embed, hidden, latent)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        logits = self.decoder(x, z)
        return logits, mu, logvar


def main():
    set_seed(SEED)

    ds = SeqDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = TextVAE(VOCAB, EMBED, HIDDEN, LATENT).to(DEVICE)
    recon_loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for xb in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, mu, logvar = model(xb)
            loss_recon = recon_loss(logits.view(-1, VOCAB), xb.view(-1))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_recon + 0.1 * kl
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
        logits, _, _ = model(sample)
        recon = torch.argmax(logits, dim=-1).cpu().tolist()[0]
    print("Input:", sample.cpu().tolist()[0])
    print("Recon:", recon)


if __name__ == "__main__":
    main()
