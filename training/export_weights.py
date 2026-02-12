"""Export trained PyTorch LISTA/ALISTA weights to binary format for C++.

Binary format (LISTA):
    Header: K (int32), n (int32), m (int32)
    Per layer k = 0..K-1:
        S[k]: n×n float64 (column-major for Eigen compatibility)
        W_e[k]: n×m float64 (column-major)
        theta[k]: n float64

Binary format (ALISTA):
    Header: K (int32)
    Per layer k = 0..K-1:
        gamma[k]: float64
        theta[k]: float64

IMPORTANT: Eigen stores matrices in column-major order by default.
    NumPy uses row-major. We transpose before saving so Eigen reads correctly.
"""

import argparse
import struct
import numpy as np
import torch


def export_lista_weights(model_path: str, output_path: str):
    """Export LISTA weights from PyTorch checkpoint to binary."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    # Count layers
    K = 0
    while f"layers.{K}.S.weight" in state:
        K += 1

    n = state["layers.0.S.weight"].shape[0]
    m = state["layers.0.W_e.weight"].shape[1]

    print(f"Exporting LISTA: K={K}, n={n}, m={m}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("iii", K, n, m))

        for k in range(K):
            # nn.Linear weight shape: (out, in). Forward: output = input @ weight.T
            # For column-vector math: output = weight @ input
            # C++ does: S * c = weight @ c, so we need Eigen S = weight
            # S.T.tobytes() writes weight in column-major → Eigen reads it as weight
            S = state[f"layers.{k}.S.weight"].numpy().astype(np.float64)
            f.write(np.ascontiguousarray(S.T).tobytes())

            W_e = state[f"layers.{k}.W_e.weight"].numpy().astype(np.float64)
            f.write(np.ascontiguousarray(W_e.T).tobytes())

            # theta
            theta = state[f"layers.{k}.threshold.theta"].numpy().astype(np.float64)
            f.write(np.abs(theta).tobytes())

    print(f"Saved to {output_path}")


def export_alista_weights(model_path: str, output_path: str):
    """Export ALISTA weights from PyTorch checkpoint to binary."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    gamma = state["gamma"].numpy().astype(np.float64)
    theta = state["theta"].numpy().astype(np.float64)
    K = len(gamma)

    print(f"Exporting ALISTA: K={K}")
    print(f"  gamma = {gamma}")
    print(f"  theta = {np.abs(theta)}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("i", K))
        for k in range(K):
            f.write(struct.pack("d", gamma[k]))
            f.write(struct.pack("d", abs(theta[k])))

    print(f"Saved to {output_path}")


def export_matrix(npy_path: str, output_path: str):
    """Export a NumPy matrix to binary format for Eigen (column-major)."""
    A = np.load(npy_path).astype(np.float64)
    m, n = A.shape
    print(f"Exporting matrix: {m}x{n}")
    with open(output_path, "wb") as f:
        f.write(struct.pack("ii", m, n))
        # Transpose for column-major (Eigen default)
        f.write(A.T.tobytes())
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export weights to binary for C++")
    parser.add_argument("model_path", help="PyTorch .pt checkpoint or .npy matrix")
    parser.add_argument("output_path", help="Output binary file")
    parser.add_argument("--type", choices=["lista", "alista", "matrix"], default="lista")
    args = parser.parse_args()

    if args.type == "lista":
        export_lista_weights(args.model_path, args.output_path)
    elif args.type == "alista":
        export_alista_weights(args.model_path, args.output_path)
    else:
        export_matrix(args.model_path, args.output_path)
