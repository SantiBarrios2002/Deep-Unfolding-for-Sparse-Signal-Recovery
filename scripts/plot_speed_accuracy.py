"""Experiment 3: Plot NMSE vs wall-clock time (speed-accuracy tradeoff)."""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def main(csv_path: str = "data/results/bench_speed.csv"):
    # Expected CSV: method, time_us, nmse_db
    methods = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["method"]
            methods[name] = (float(row["time_us"]), float(row["nmse_db"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"ISTA": "blue", "FISTA": "green", "LISTA": "red", "ALISTA": "magenta"}
    markers = {"ISTA": "o", "FISTA": "s", "LISTA": "^", "ALISTA": "D"}

    for name, (time_us, nmse) in methods.items():
        ax.scatter(time_us, nmse, c=colors.get(name, "gray"),
                   marker=markers.get(name, "o"), s=120, label=name, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (us)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Speed vs Accuracy: C++ Inference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("data/results/speed_accuracy.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/results/bench_speed.csv"
    main(path)
