"""Train ALISTA network (Liu et al., ICLR 2019) using PyTorch.

Key difference from LISTA: weight matrices are computed analytically from A.
Only step sizes (gamma) and thresholds (theta) are learned — 2 params per layer.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_generator import generate_dataset


class ALISTANetwork(nn.Module):
    """ALISTA: Analytic LISTA with learned step sizes and thresholds only."""

    def __init__(self, n: int, m: int, K: int, A: np.ndarray, alpha: float = 1e-3):
        super().__init__()
        self.K = K
        self.n = n

        # Register analytic matrices as buffers (not trainable)
        A_t = torch.from_numpy(A).float()
        AtA = A_t.T @ A_t
        reg = AtA + alpha * torch.eye(n)
        W = torch.linalg.solve(reg, A_t.T)

        self.register_buffer("A", A_t)
        self.register_buffer("W", W)

        # Learnable parameters: gamma (step size) and theta (threshold) per layer
        self.gamma = nn.Parameter(torch.ones(K))
        self.theta = nn.Parameter(torch.ones(K) * 0.1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch_size = y.size(0)
        c = torch.zeros(batch_size, self.n, device=y.device)

        for k in range(self.K):
            residual = c @ self.A.T - y  # (batch, m)
            z = c - self.gamma[k] * (residual @ self.W.T)
            c = torch.sign(z) * torch.relu(torch.abs(z) - torch.abs(self.theta[k]))

        return c


def train(args):
    print("Generating training data...")
    A, Y_train, X_train = generate_dataset(
        m=args.m, n=args.n, k=args.k,
        num_samples=args.num_train, snr_db=args.snr_db, seed=42,
    )
    _, Y_val, X_val = generate_dataset(
        m=args.m, n=args.n, k=args.k,
        num_samples=args.num_val, snr_db=args.snr_db, seed=123,
    )

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ALISTANetwork(args.n, args.m, args.K, A, alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Training ALISTA: n={args.n}, m={args.m}, K={args.K}")
    print(f"Total trainable parameters: {total_params}")
    print(f"Device: {device}")

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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_batch, x_batch in val_loader:
                y_batch, x_batch = y_batch.to(device), x_batch.to(device)
                x_hat = model(y_batch)
                val_loss += criterion(x_hat, x_batch).item() * y_batch.size(0)
        val_loss /= len(val_ds)

        nmse_db = 10 * np.log10(val_loss / np.mean(X_val**2) + 1e-15)

        # Print current gamma and theta values
        gamma_str = ", ".join(f"{g:.3f}" for g in model.gamma.data.cpu().numpy())
        theta_str = ", ".join(f"{t:.4f}" for t in model.theta.data.cpu().numpy())
        print(f"Epoch {epoch+1:3d}/{args.epochs}: "
              f"train={train_loss:.6f}, val={val_loss:.6f}, NMSE={nmse_db:.1f} dB")

    torch.save(model.state_dict(), args.output)
    np.save(args.output.replace(".pt", "_A.npy"), A)
    print(f"\nModel saved to {args.output}")
    print(f"Learned gamma: {model.gamma.data.cpu().numpy()}")
    print(f"Learned theta: {model.theta.data.cpu().numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALISTA network")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=250)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--output", type=str, default="alista_weights.pt")
    args = parser.parse_args()
    train(args)
