"""Visualization utilities for training and evaluation."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
):
    """Plot training (and optionally validation) loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, "b-", label="Train")
    if val_losses:
        ax.semilogy(epochs, val_losses, "r--", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_nmse_vs_iterations(
    results: dict,
    title: str = "NMSE vs Iterations/Layers",
    save_path: Optional[str] = None,
):
    """
    Plot NMSE convergence for multiple solvers.

    Args:
        results: dict mapping solver_name → (iterations_array, nmse_array)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v"]
    for i, (name, (iters, nmse)) in enumerate(results.items()):
        ax.plot(iters, nmse, marker=markers[i % len(markers)],
                label=name, markersize=4)
    ax.set_xlabel("Iterations / Layers")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_doa_spectrum(
    theta_grid_deg: np.ndarray,
    spectra: dict,
    true_doas: List[float],
    title: str = "DOA Spatial Spectrum",
    save_path: Optional[str] = None,
):
    """
    Plot angular spectra for multiple methods.

    Args:
        theta_grid_deg: angle grid in degrees
        spectra: dict mapping method_name → spectrum_array
        true_doas: list of true DOA angles in degrees
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, spectrum in spectra.items():
        spectrum_db = 10 * np.log10(spectrum / spectrum.max() + 1e-15)
        ax.plot(theta_grid_deg, spectrum_db, label=name)

    for doa in true_doas:
        ax.axvline(x=doa, color="k", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Normalized Spectrum (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
