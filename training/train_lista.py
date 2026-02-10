"""Train LISTA network (Gregor & LeCun 2010) using PyTorch."""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_generator import generate_dataset


class SoftThreshold(nn.Module):
    """Learnable soft-thresholding activation."""

    def __init__(self, n: int):
        super().__init__()
        self.theta = nn.Parameter(torch.ones(n) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - torch.abs(self.theta))


class LISTALayer(nn.Module):
    """Single LISTA layer: c = soft_threshold(S @ c_prev + W_e @ y, theta)."""

    def __init__(self, n: int, m: int):
        super().__init__()
        self.S = nn.Linear(n, n, bias=False)
        self.W_e = nn.Linear(m, n, bias=False)
        self.threshold = SoftThreshold(n)

    def forward(self, c_prev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.threshold(self.S(c_prev) + self.W_e(y))


class LISTANetwork(nn.Module):
    """LISTA: K unfolded layers with untied parameters."""

    def __init__(self, n: int, m: int, K: int, A: np.ndarray):
        super().__init__()
        self.K = K
        self.layers = nn.ModuleList([LISTALayer(n, m) for _ in range(K)])

        # Initialize from ISTA parameters
        A_t = torch.from_numpy(A).float()
        AtA = A_t.T @ A_t
        L = torch.linalg.eigvalsh(AtA).max().item()

        S_init = torch.eye(n) - (1.0 / L) * AtA
        W_e_init = (1.0 / L) * A_t.T

        for layer in self.layers:
            layer.S.weight.data.copy_(S_init)
            layer.W_e.weight.data.copy_(W_e_init)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        c = self.layers[0].W_e(y)
        c = self.layers[0].threshold(c)

        for k in range(1, self.K):
            c = self.layers[k](c, y)

        return c


def train(args):
    # Generate data
    print("Generating training data...")
    A, Y_train, X_train = generate_dataset(
        m=args.m, n=args.n, k=args.k,
        num_samples=args.num_train, snr_db=args.snr_db, seed=42,
    )
    _, Y_val, X_val = generate_dataset(
        m=args.m, n=args.n, k=args.k,
        num_samples=args.num_val, snr_db=args.snr_db, seed=123,
    )

    # Create datasets
    train_ds = TensorDataset(
        torch.from_numpy(Y_train).float(),
        torch.from_numpy(X_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(Y_val).float(),
        torch.from_numpy(X_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LISTANetwork(args.n, args.m, args.K, A).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Training LISTA: n={args.n}, m={args.m}, K={args.K}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for y_batch, x_batch in train_loader:
            y_batch, x_batch = y_batch.to(device), x_batch.to(device)
            x_hat = model(y_batch)
            loss = criterion(x_hat, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_batch, x_batch in val_loader:
                y_batch, x_batch = y_batch.to(device), x_batch.to(device)
                x_hat = model(y_batch)
                val_loss += criterion(x_hat, x_batch).item() * y_batch.size(0)
        val_loss /= len(val_ds)

        nmse_db = 10 * np.log10(val_loss / np.mean(X_val**2) + 1e-15)
        print(f"Epoch {epoch+1:3d}/{args.epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
              f"NMSE={nmse_db:.1f} dB")

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"\nModel saved to {args.output}")

    # Also save A for export
    np.save(args.output.replace(".pt", "_A.npy"), A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LISTA network")
    parser.add_argument("--n", type=int, default=500, help="Signal dimension")
    parser.add_argument("--m", type=int, default=250, help="Measurement dimension")
    parser.add_argument("--k", type=int, default=25, help="Sparsity level")
    parser.add_argument("--K", type=int, default=16, help="Number of LISTA layers")
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="lista_weights.pt")
    args = parser.parse_args()
    train(args)
