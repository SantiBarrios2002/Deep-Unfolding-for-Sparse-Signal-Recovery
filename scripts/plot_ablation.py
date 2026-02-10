"""Experiment 5: Plot ablation study results."""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def plot_sparsity_generalization(csv_path: str):
    """5a: NMSE vs sparsity level for models trained on k=25."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for col in columns:
            data[col] = []
        for row in reader:
            for col in columns:
                data[col].append(float(row[col]))

    k_values = np.array(data["sparsity"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in columns:
        if col == "sparsity":
            continue
        ax.plot(k_values, data[col], "-o", label=col)

    ax.axvline(x=25, color="k", linestyle="--", alpha=0.3, label="Training k=25")
    ax.set_xlabel("Sparsity k")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Sparsity Generalization (trained on k=25)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/results/ablation_sparsity.png", dpi=150)


def plot_layer_sweep(csv_path: str):
    """5b: NMSE vs number of layers K."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for col in columns:
            data[col] = []
        for row in reader:
            for col in columns:
                data[col].append(float(row[col]))

    K_values = np.array(data["layers"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in columns:
        if col == "layers":
            continue
        ax.plot(K_values, data[col], "-o", label=col)

    ax.set_xlabel("Number of Layers K")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("NMSE vs Layer Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/results/ablation_layers.png", dpi=150)


def plot_parameter_count():
    """5c: Bar chart of trainable parameters."""
    n, m, K = 500, 250, 16
    lista_params = K * (n * n + n * m + n)
    alista_params = K * 2

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["LISTA", "ALISTA"], [lista_params, alista_params],
                   color=["red", "magenta"])
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Parameter Count Comparison (n=500, K=16)")
    ax.set_yscale("log")

    for bar, val in zip(bars, [lista_params, alista_params]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:,}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("data/results/ablation_params.png", dpi=150)


def main():
    plot_parameter_count()
    # Other plots require CSV data from experiments
    plt.show()


if __name__ == "__main__":
    main()
