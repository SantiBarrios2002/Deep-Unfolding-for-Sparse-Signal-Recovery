"""Experiment 2: Plot DOA RMSE vs SNR for multiple methods."""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def main(csv_path: str = "data/results/doa_rmse_vs_snr.csv"):
    # Expected CSV: snr_db, then one column per method (RMSE values)
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for col in columns:
            data[col] = []
        for row in reader:
            for col in columns:
                data[col].append(float(row[col]))

    snr = np.array(data["snr_db"])

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v"]
    methods = [c for c in columns if c != "snr_db"]

    for i, method in enumerate(methods):
        rmse = np.array(data[method])
        ax.semilogy(snr, rmse, marker=markers[i % len(markers)],
                     label=method, markersize=6)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("DOA RMSE (degrees)")
    ax.set_title("DOA Estimation: RMSE vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/results/doa_rmse_vs_snr.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/results/doa_rmse_vs_snr.csv"
    main(path)
