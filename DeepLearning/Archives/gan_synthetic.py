import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------
# Simple config
# ----------------------
SEED = 101
DEVICE = "cpu"  # change to "cuda" if available
EPOCHS = 1000
BATCH_SIZE = 64
LATENT_DIM = 128
HIDDEN = [64, 32, 32, 32, 8]
LR = 1e-4
BETA1 = 0.5
N_SYNTHETIC = 500


class Scaler:
    def __init__(self, min_: np.ndarray, max_: np.ndarray, columns: list[str]) -> None:
        self.min_ = min_
        self.max_ = max_
        self.columns = columns

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = (self.max_ - self.min_)
        denom[denom == 0] = 1.0
        return (x - self.min_) / denom

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.max_ - self.min_) + self.min_


def fit_minmax(df: pd.DataFrame) -> Scaler:
    min_ = df.min(axis=0).to_numpy(dtype=np.float32)
    max_ = df.max(axis=0).to_numpy(dtype=np.float32)
    return Scaler(min_=min_, max_=max_, columns=list(df.columns))


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: list[int], p_dropout: float = 0.5):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], p_dropout: float = 0.5):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def sample_generator(generator: nn.Module, n: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    z = torch.randn(n, latent_dim, device=device)
    generator.eval()
    return generator(z)


def train_gan(
    data: torch.Tensor,
    latent_dim: int,
    hidden: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    beta1: float,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    generator = Generator(latent_dim, data.shape[1], hidden).to(device)
    discriminator = Discriminator(data.shape[1], hidden).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    bce = nn.BCELoss()

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_len = real_batch.size(0)

            # Train discriminator
            z = torch.randn(batch_len, latent_dim, device=device)
            fake_batch = generator(z).detach()
            real_labels = torch.ones(batch_len, 1, device=device)
            fake_labels = torch.zeros(batch_len, 1, device=device)

            d_real = discriminator(real_batch)
            d_fake = discriminator(fake_batch)
            d_loss = 0.5 * (bce(d_real, real_labels) + bce(d_fake, fake_labels))

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train generator
            z = torch.randn(batch_len, latent_dim, device=device)
            gen_batch = generator(z)
            g_loss = bce(discriminator(gen_batch), real_labels)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:05d}  D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}")

    return generator, discriminator


def create_demo_dataset(seed: int) -> pd.DataFrame:
    # Small imbalanced dataset with numeric features + label
    n_major = 1000
    n_minor = 50
    rng = np.random.default_rng(seed)

    maj = rng.normal(loc=0.0, scale=1.0, size=(n_major, 6))
    minr = rng.normal(loc=1.5, scale=1.2, size=(n_minor, 6))

    data = np.vstack([maj, minr]).astype(np.float32)
    labels = np.array([0] * n_major + [1] * n_minor, dtype=np.int64)

    cols = [f"f{i+1}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    df["label"] = labels
    return df


def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")

    # 1. Create an imbalanced demo dataset
    df = create_demo_dataset(SEED)
    print("Demo dataset created.")
    print(f"Total rows: {df.shape[0]}, features: {df.shape[1] - 1}")
    print("Class distribution:")
    print(df["label"].value_counts().to_string())

    # 2. Train GAN only on the minority class (label=1)
    fraud_df = df[df["label"] == 1].drop(columns=["label"]).copy()

    scaler = fit_minmax(fraud_df)

    data = scaler.transform(fraud_df.to_numpy(dtype=np.float32))
    data_t = torch.from_numpy(data).float()

    generator, _ = train_gan(
        data=data_t,
        latent_dim=LATENT_DIM,
        hidden=HIDDEN,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        beta1=BETA1,
        device=device,
    )

    # 3. Generate synthetic minority samples
    synth = sample_generator(generator, N_SYNTHETIC, LATENT_DIM, device).cpu().numpy()
    synth = scaler.inverse_transform(synth)

    synth_df = pd.DataFrame(synth, columns=fraud_df.columns)
    synth_df["label"] = 1
    print(f"Synthetic samples generated: {len(synth_df)}")
    print("Synthetic data preview:")
    print(synth_df.head().to_string(index=False))

    # 4. Combine to create a more balanced dataset
    balanced = pd.concat([df, synth_df], ignore_index=True).sample(frac=1, random_state=SEED)
    print(f"Balanced dataset shape: {balanced.shape}")
    print("Balanced class distribution:")
    print(balanced["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
