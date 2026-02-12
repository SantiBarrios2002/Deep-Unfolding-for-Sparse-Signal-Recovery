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

    def forward(self, y: torch.Tensor, K_active: int = None) -> torch.Tensor:
        if K_active is None:
            K_active = self.K

        c = self.layers[0].W_e(y)
        c = self.layers[0].threshold(c)

        for k in range(1, K_active):
            c = self.layers[k](c, y)

        return c


def build_stage_schedule(K, total_epochs, base_lr):
    """Build progressive training stages: double K_active each stage.

    Returns list of (K_active, epochs, lr) tuples.
    """
    # K_active values: powers of 2 up to K
    k_values = []
    k = 2
    while k < K:
        k_values.append(k)
        k *= 2
    k_values.append(K)

    num_stages = len(k_values)

    # Epoch allocation: weighted toward earlier (shallower) stages
    weights = np.linspace(3.0, 2.0, num_stages)
    weights /= weights.sum()
    epoch_alloc = [max(5, int(round(w * total_epochs))) for w in weights]
    # Adjust last stage to hit total_epochs exactly
    epoch_alloc[-1] = max(5, total_epochs - sum(epoch_alloc[:-1]))

    # LR schedule: decrease 0.6x per stage
    lr_schedule = [base_lr * (0.6 ** i) for i in range(num_stages)]

    return list(zip(k_values, epoch_alloc, lr_schedule))


def train_stage(model, train_loader, val_loader, K_active, epochs, lr,
                max_norm, device, sig_power, best_val_loss, best_state):
    """Train a single stage (fixed K_active depth).

    Returns (best_val_loss, best_state) updated across this stage.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.MSELoss()
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for y_batch, x_batch in train_loader:
            y_batch, x_batch = y_batch.to(device), x_batch.to(device)
            x_hat = model(y_batch, K_active=K_active)
            loss = criterion(x_hat, x_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)

        train_loss /= n_train
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for y_batch, x_batch in val_loader:
                y_batch, x_batch = y_batch.to(device), x_batch.to(device)
                x_hat = model(y_batch, K_active=K_active)
                val_loss += criterion(x_hat, x_batch).item() * y_batch.size(0)
        val_loss /= n_val

        nmse_db = 10 * np.log10(val_loss / sig_power + 1e-15)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:3d}/{epochs}: "
              f"train={train_loss:.6f}, val={val_loss:.6f}, "
              f"NMSE={nmse_db:.1f} dB, lr={lr_now:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_loss, best_state


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
        A=A,  # use same sensing matrix as training
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

    print(f"Training LISTA: n={args.n}, m={args.m}, K={args.K}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    sig_power = np.mean(X_val**2)
    best_val_loss = float("inf")
    best_state = None

    if args.progressive:
        # Progressive training: train shallow first, then deepen
        if args.stages:
            k_values = [int(x) for x in args.stages.split(",")]
            num_stages = len(k_values)
            weights = np.linspace(3.0, 2.0, num_stages)
            weights /= weights.sum()
            epoch_alloc = [max(5, int(round(w * args.epochs))) for w in weights]
            epoch_alloc[-1] = max(5, args.epochs - sum(epoch_alloc[:-1]))
            lr_schedule = [args.lr * (0.6 ** i) for i in range(num_stages)]
            stages = list(zip(k_values, epoch_alloc, lr_schedule))
        else:
            stages = build_stage_schedule(args.K, args.epochs, args.lr)

        print(f"\nProgressive training: {len(stages)} stages")
        for i, (ka, ep, lr) in enumerate(stages):
            print(f"  Stage {i+1}: K_active={ka}, epochs={ep}, lr={lr:.1e}")

        for stage_idx, (K_active, stage_epochs, stage_lr) in enumerate(stages):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx+1}/{len(stages)}: "
                  f"K_active={K_active}, epochs={stage_epochs}, lr={stage_lr:.1e}")
            print(f"{'='*60}")

            # Restore best checkpoint from previous stage
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(device)

            best_val_loss, best_state = train_stage(
                model, train_loader, val_loader,
                K_active, stage_epochs, stage_lr,
                max_norm=5.0, device=device, sig_power=sig_power,
                best_val_loss=best_val_loss, best_state=best_state,
            )

            stage_nmse = 10 * np.log10(best_val_loss / sig_power + 1e-15)
            print(f"Stage {stage_idx+1} best NMSE: {stage_nmse:.1f} dB")
    else:
        # End-to-end training (original behavior)
        print(f"\nEnd-to-end training: K={args.K}, epochs={args.epochs}")
        best_val_loss, best_state = train_stage(
            model, train_loader, val_loader,
            args.K, args.epochs, args.lr,
            max_norm=10.0, device=device, sig_power=sig_power,
            best_val_loss=best_val_loss, best_state=best_state,
        )

    # Save best model
    best_nmse = 10 * np.log10(best_val_loss / sig_power + 1e-15)
    print(f"\nBest validation NMSE: {best_nmse:.1f} dB")
    torch.save(best_state, args.output)
    print(f"Model saved to {args.output}")

    # Also save A for export
    np.save(args.output.replace(".pt", "_A.npy"), A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LISTA network")
    parser.add_argument("--n", type=int, default=500, help="Signal dimension")
    parser.add_argument("--m", type=int, default=250, help="Measurement dimension")
    parser.add_argument("--k", type=int, default=25, help="Sparsity level")
    parser.add_argument("--K", type=int, default=16, help="Number of LISTA layers")
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--num-train", type=int, default=50000)
    parser.add_argument("--num-val", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--progressive", action="store_true",
                        help="Use progressive training: shallow layers first, "
                             "then gradually increase depth")
    parser.add_argument("--stages", type=str, default=None,
                        help="Custom K_active schedule, e.g. '2,4,8,16'")
    parser.add_argument("--output", type=str, default="lista_weights.pt")
    args = parser.parse_args()
    train(args)
