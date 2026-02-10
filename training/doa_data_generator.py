"""Generate DOA estimation data for training LISTA/ALISTA on sparse recovery."""

import numpy as np
from typing import List, Tuple


def build_steering_matrix(
    M: int, d_lambda: float, theta_grid: np.ndarray
) -> np.ndarray:
    """Build steering matrix D ∈ ℂ^{M×N} for a ULA."""
    N = len(theta_grid)
    D = np.zeros((M, N), dtype=complex)
    for j in range(N):
        for i in range(M):
            phase = 2.0 * np.pi * i * d_lambda * np.sin(theta_grid[j])
            D[i, j] = np.exp(1j * phase)
    return D


def complex_to_real_stacked_vec(y: np.ndarray) -> np.ndarray:
    """Stack complex vector: [Re(y); Im(y)]."""
    return np.concatenate([y.real, y.imag])


def complex_to_real_stacked_mat(D: np.ndarray) -> np.ndarray:
    """Stack complex matrix: [Re(D), -Im(D); Im(D), Re(D)]."""
    M, N = D.shape
    A_real = np.block([
        [D.real, -D.imag],
        [D.imag, D.real],
    ])
    return A_real


def generate_doa_dataset(
    M: int = 16,
    d_lambda: float = 0.5,
    grid_points: int = 181,
    num_sources: int = 3,
    num_samples: int = 10000,
    snr_db: float = 20.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate DOA training data as real-valued stacked sparse recovery problems.

    Returns:
        A_real: real-valued stacked steering matrix (2M, 2N)
        Y: measurements (num_samples, 2M)
        X: sparse ground truth (num_samples, 2N) — nonzeros at true DOA grid indices
    """
    rng = np.random.default_rng(seed)

    theta_grid_deg = np.linspace(-90, 90, grid_points)
    theta_grid_rad = np.deg2rad(theta_grid_deg)

    D = build_steering_matrix(M, d_lambda, theta_grid_rad)
    A_real = complex_to_real_stacked_mat(D)
    N = grid_points

    Y = np.zeros((num_samples, 2 * M))
    X = np.zeros((num_samples, 2 * N))

    for i in range(num_samples):
        # Random DOA angles (uniformly from grid)
        doa_indices = rng.choice(N, size=num_sources, replace=False)

        # Complex source signals
        s = (rng.normal(0, 1, num_sources) + 1j * rng.normal(0, 1, num_sources)) / np.sqrt(2)

        # Build ground truth sparse vector (complex)
        x_complex = np.zeros(N, dtype=complex)
        x_complex[doa_indices] = s

        # Stack to real
        x_real = np.concatenate([x_complex.real, x_complex.imag])

        # Measurement
        y_complex = D @ x_complex

        # Add noise
        noise = (rng.normal(0, 1, M) + 1j * rng.normal(0, 1, M)) / np.sqrt(2)
        signal_power = np.mean(np.abs(y_complex) ** 2)
        noise_power = signal_power * 10 ** (-snr_db / 10)
        noise *= np.sqrt(noise_power / np.maximum(np.mean(np.abs(noise) ** 2), 1e-15))
        y_complex += noise

        y_real = complex_to_real_stacked_vec(y_complex)

        Y[i] = y_real
        X[i] = x_real

    return A_real, Y, X


if __name__ == "__main__":
    A, Y, X = generate_doa_dataset(num_samples=100)
    print(f"A: {A.shape}, Y: {Y.shape}, X: {X.shape}")
