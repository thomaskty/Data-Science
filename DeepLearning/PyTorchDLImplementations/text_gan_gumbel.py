#!/usr/bin/env python3
"""Simple text GAN with Gumbel-Softmax (educational).

- Generator produces token distributions for fixed-length sequences
- Discriminator classifies real vs fake sequences
- Synthetic real data = increasing sequences modulo vocab

Note: Text GANs are hard; this is a minimal teaching example.
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
EPOCHS = 30
LR = 1e-3
EMBED = 32
HIDDEN = 64
GUMBEL_TEMP = 0.8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RealDataset(Dataset):
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


class Generator(nn.Module):
    def __init__(self, vocab, seq_len, hidden):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, seq_len * vocab)
        )
        self.vocab = vocab
        self.seq_len = seq_len

    def forward(self, z):
        logits = self.fc(z).view(-1, self.seq_len, self.vocab)
        return logits


class Discriminator(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.rnn(emb)
        logits = self.fc(h.squeeze(0))
        return logits


def gumbel_softmax(logits, temperature):
    y = torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
    return y


def sample_tokens(logits):
    probs = torch.softmax(logits, dim=-1)
    return torch.argmax(probs, dim=-1)


def main():
    set_seed(SEED)

    ds = RealDataset()
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    G = Generator(VOCAB, SEQ_LEN, HIDDEN).to(DEVICE)
    D = Discriminator(VOCAB, EMBED, HIDDEN).to(DEVICE)

    opt_g = torch.optim.Adam(G.parameters(), lr=LR)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        g_loss_total = 0.0
        d_loss_total = 0.0
        count = 0

        for real in dl:
            real = real.to(DEVICE)
            bs = real.size(0)
            count += bs

            # Discriminator step
            z = torch.randn(bs, HIDDEN).to(DEVICE)
            logits = G(z)
            fake_tokens = sample_tokens(logits).detach()

            real_logits = D(real)
            fake_logits = D(fake_tokens)

            d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(
                fake_logits, torch.zeros_like(fake_logits)
            )
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # Generator step (use Gumbel-Softmax for differentiable tokens)
            z = torch.randn(bs, HIDDEN).to(DEVICE)
            logits = G(z)
            y = gumbel_softmax(logits, GUMBEL_TEMP)  # [bs, seq, vocab]
            # Convert soft one-hot to embedding-like indices via argmax for D input
            fake_tokens = torch.argmax(y, dim=-1)

            g_logits = D(fake_tokens)
            g_loss = bce(g_logits, torch.ones_like(g_logits))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            g_loss_total += g_loss.item() * bs
            d_loss_total += d_loss.item() * bs

        print(
            f"Epoch {epoch:02d} | D loss: {d_loss_total / count:.4f} | G loss: {g_loss_total / count:.4f}"
        )

    # Sample generation
    G.eval()
    with torch.no_grad():
        z = torch.randn(1, HIDDEN).to(DEVICE)
        logits = G(z)
        tokens = sample_tokens(logits).cpu().tolist()[0]
    print("Generated sequence:", tokens)


if __name__ == "__main__":
    main()
