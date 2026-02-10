"""Experiment 2: Plot DOA angular spectrum for multiple methods."""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def main(csv_path: str = "data/results/doa_spectrum.csv"):
    # Expected CSV: angle_deg, then one column per method
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for col in columns:
            data[col] = []
        for row in reader:
            for col in columns:
                data[col].append(float(row[col]))

    theta = np.array(data["angle_deg"])
    true_doas = [-30, 10, 45]

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in columns:
        if col == "angle_deg":
            continue
        spectrum = np.array(data[col])
        spectrum_db = 10 * np.log10(spectrum / spectrum.max() + 1e-15)
        ax.plot(theta, spectrum_db, label=col)

    for doa in true_doas:
        ax.axvline(x=doa, color="k", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Normalized Spectrum (dB)")
    ax.set_title("DOA Estimation: Angular Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    plt.tight_layout()
    plt.savefig("data/results/doa_spectrum.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/results/doa_spectrum.csv"
    main(path)
