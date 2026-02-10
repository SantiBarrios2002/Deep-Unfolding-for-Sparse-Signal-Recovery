"""Experiment 1: Plot NMSE vs iterations/layers for all solvers."""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def main(csv_path: str = "data/results/bench_synthetic.csv"):
    # Expected CSV columns: method, iteration_or_layer, nmse_db
    methods = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["method"]
            if name not in methods:
                methods[name] = ([], [])
            methods[name][0].append(int(row["iteration_or_layer"]))
            methods[name][1].append(float(row["nmse_db"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {"ISTA": "b-o", "FISTA": "g-s", "LISTA": "r-^", "ALISTA": "m-D"}

    for name, (iters, nmse) in methods.items():
        style = styles.get(name, "-")
        ax.plot(iters, nmse, style, label=name, markersize=4)

    ax.set_xlabel("Iterations / Layers")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Sparse Recovery: NMSE vs Iterations/Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/results/nmse_vs_iterations.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/results/bench_synthetic.csv"
    main(path)
