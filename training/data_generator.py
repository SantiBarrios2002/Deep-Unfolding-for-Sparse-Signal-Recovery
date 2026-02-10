"""Generate synthetic compressed sensing (A, x*, y) pairs for training."""

import numpy as np
from typing import Tuple


def generate_sensing_matrix(m: int, n: int, seed: int = 42) -> np.ndarray:
    """Generate an m×n Gaussian random matrix with normalized columns."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1.0 / np.sqrt(m), size=(m, n))
    # Normalize columns to unit ℓ₂ norm
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    A = A / np.maximum(norms, 1e-15)
    return A


def generate_sparse_signal(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a k-sparse signal of dimension n with N(0,1) nonzeros."""
    x = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    x[support] = rng.normal(0, 1, size=k)
    return x


def generate_dataset(
    m: int = 250,
    n: int = 500,
    k: int = 25,
    num_samples: int = 10000,
    snr_db: float = 40.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dataset of (y, x*) pairs for training LISTA/ALISTA.

    Returns:
        A: sensing matrix (m, n)
        Y: measurement matrix (num_samples, m)
        X: ground truth sparse signals (num_samples, n)
    """
    rng = np.random.default_rng(seed)
    A = generate_sensing_matrix(m, n, seed)

    X = np.zeros((num_samples, n))
    Y = np.zeros((num_samples, m))

    for i in range(num_samples):
        x = generate_sparse_signal(n, k, rng)
        y_clean = A @ x

        # Add noise at specified SNR
        noise = rng.normal(0, 1, size=m)
        signal_power = np.mean(y_clean**2)
        noise_power = signal_power * 10 ** (-snr_db / 10)
        noise *= np.sqrt(noise_power / np.maximum(np.mean(noise**2), 1e-15))
        y = y_clean + noise

        X[i] = x
        Y[i] = y

    return A, Y, X


if __name__ == "__main__":
    A, Y, X = generate_dataset(num_samples=1000)
    print(f"A: {A.shape}, Y: {Y.shape}, X: {X.shape}")
    print(f"Sparsity check: avg nonzeros = {np.mean(np.sum(X != 0, axis=1)):.1f}")
