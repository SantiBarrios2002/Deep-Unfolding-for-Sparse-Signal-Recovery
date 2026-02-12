"""Debug LISTA: compare Python model vs exported binary weights vs C++ logic."""
import struct
import numpy as np
import torch
from train_lista import LISTANetwork
from data_generator import generate_dataset


def load_lista_bin(path):
    """Load lista.bin the same way C++ does (column-major Eigen convention)."""
    with open(path, "rb") as f:
        K, n, m = struct.unpack("iii", f.read(12))
        print(f"Binary header: K={K}, n={n}, m={m}")
        layers = []
        for k in range(K):
            # Eigen reads column-major: n*n doubles
            S_data = np.frombuffer(f.read(n * n * 8), dtype=np.float64)
            S = S_data.reshape((n, n), order="F")  # column-major → Eigen convention

            W_data = np.frombuffer(f.read(n * m * 8), dtype=np.float64)
            W_e = W_data.reshape((n, m), order="F")  # column-major

            theta = np.frombuffer(f.read(n * 8), dtype=np.float64).copy()
            layers.append({"S": S, "W_e": W_e, "theta": theta})
        return K, n, m, layers


def soft_threshold_np(u, alpha):
    """Same as C++ soft_threshold."""
    return np.sign(u) * np.maximum(np.abs(u) - np.abs(alpha), 0.0)


def lista_forward_cpp_style(y, layers, K):
    """Simulate the C++ forward pass: column-vector math."""
    # Layer 0: c = soft_threshold(W_e[0] * y, theta[0])
    c = soft_threshold_np(layers[0]["W_e"] @ y, layers[0]["theta"])

    # Layers 1..K-1: c = soft_threshold(S[k]*c + W_e[k]*y, theta[k])
    for k in range(1, K):
        c = soft_threshold_np(
            layers[k]["S"] @ c + layers[k]["W_e"] @ y, layers[k]["theta"]
        )
    return c


def main():
    # 1. Load trained model
    print("=" * 60)
    print("1. Loading PyTorch model")
    A, Y_val, X_val = generate_dataset(
        m=250, n=500, k=25, num_samples=100, snr_db=40.0, seed=999
    )
    model = LISTANetwork(500, 250, 16, A)
    state = torch.load("lista_weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 2. Python model inference
    print("\n2. Python model inference")
    with torch.no_grad():
        y_t = torch.from_numpy(Y_val).float()
        x_t = torch.from_numpy(X_val).float()
        x_hat_py = model(y_t).numpy()

    mse_py = np.mean((x_hat_py - X_val) ** 2)
    sig_py = np.mean(X_val**2)
    nmse_py = 10 * np.log10(mse_py / sig_py + 1e-15)
    print(f"   Python LISTA NMSE = {nmse_py:.2f} dB")

    # 3. Load exported binary
    print("\n3. Loading lista.bin")
    K, n, m, bin_layers = load_lista_bin("../data/weights/lista.bin")

    # 4. Compare weights: state_dict vs binary
    print("\n4. Comparing weights (state_dict vs binary)")
    max_err_S = 0
    max_err_W = 0
    max_err_t = 0
    for k in range(K):
        S_pt = state[f"layers.{k}.S.weight"].numpy().astype(np.float64)
        W_pt = state[f"layers.{k}.W_e.weight"].numpy().astype(np.float64)
        t_pt = np.abs(state[f"layers.{k}.threshold.theta"].numpy().astype(np.float64))

        S_bin = bin_layers[k]["S"]
        W_bin = bin_layers[k]["W_e"]
        t_bin = bin_layers[k]["theta"]

        err_S = np.max(np.abs(S_pt - S_bin))
        err_W = np.max(np.abs(W_pt - W_bin))
        err_t = np.max(np.abs(t_pt - t_bin))

        max_err_S = max(max_err_S, err_S)
        max_err_W = max(max_err_W, err_W)
        max_err_t = max(max_err_t, err_t)

        if err_S > 1e-6 or err_W > 1e-6 or err_t > 1e-6:
            print(f"   Layer {k}: S err={err_S:.2e}, W_e err={err_W:.2e}, theta err={err_t:.2e} *** MISMATCH ***")

    print(f"   Max errors: S={max_err_S:.2e}, W_e={max_err_W:.2e}, theta={max_err_t:.2e}")
    if max_err_S < 1e-6 and max_err_W < 1e-6 and max_err_t < 1e-6:
        print("   All weights match! (export is correct)")
    else:
        print("   *** WEIGHTS DO NOT MATCH — export bug! ***")

    # 5. Run C++-style forward pass using binary weights
    print("\n5. C++-style forward pass (numpy, simulating Eigen)")
    nmse_cpp_vals = []
    for i in range(min(100, len(Y_val))):
        y_vec = Y_val[i]  # shape (m,)
        x_true = X_val[i]  # shape (n,)
        x_hat = lista_forward_cpp_style(y_vec, bin_layers, K)
        err = np.sum((x_hat - x_true) ** 2)
        sig = np.sum(x_true**2)
        if sig > 1e-15:
            nmse_cpp_vals.append(err / sig)

    avg_nmse_cpp = 10 * np.log10(np.mean(nmse_cpp_vals) + 1e-15)
    print(f"   C++-style LISTA NMSE = {avg_nmse_cpp:.2f} dB")

    # 6. Also compare sample-by-sample with Python model
    print("\n6. Comparing Python model vs C++-style forward (sample 0)")
    y0 = Y_val[0]
    x_hat_cpp0 = lista_forward_cpp_style(y0, bin_layers, K)

    with torch.no_grad():
        x_hat_py0 = model(torch.from_numpy(y0).float().unsqueeze(0)).numpy()[0]

    diff = np.max(np.abs(x_hat_cpp0 - x_hat_py0.astype(np.float64)))
    print(f"   Max output diff (py vs cpp-style): {diff:.2e}")
    print(f"   py output norm: {np.linalg.norm(x_hat_py0):.4f}")
    print(f"   cpp-style output norm: {np.linalg.norm(x_hat_cpp0):.4f}")

    # 7. Step through layer by layer to find divergence point
    print("\n7. Layer-by-layer trace (sample 0)")
    y0_t = torch.from_numpy(y0).float()

    # Python model layer 0
    with torch.no_grad():
        c_py = model.layers[0].W_e(y0_t.unsqueeze(0))
        c_py = model.layers[0].threshold(c_py)
        c_py_np = c_py.numpy()[0].astype(np.float64)

    # C++-style layer 0
    c_cpp = soft_threshold_np(bin_layers[0]["W_e"] @ y0, bin_layers[0]["theta"])

    diff0 = np.max(np.abs(c_py_np - c_cpp))
    print(f"   Layer 0 max diff: {diff0:.2e}, py norm={np.linalg.norm(c_py_np):.4f}, cpp norm={np.linalg.norm(c_cpp):.4f}")

    for k in range(1, K):
        with torch.no_grad():
            c_py = model.layers[k](c_py, y0_t.unsqueeze(0))
            c_py_np = c_py.numpy()[0].astype(np.float64)

        c_cpp = soft_threshold_np(
            bin_layers[k]["S"] @ c_cpp + bin_layers[k]["W_e"] @ y0,
            bin_layers[k]["theta"],
        )
        diff_k = np.max(np.abs(c_py_np - c_cpp))
        print(f"   Layer {k:2d} max diff: {diff_k:.2e}, py norm={np.linalg.norm(c_py_np):.4f}, cpp norm={np.linalg.norm(c_cpp):.4f}")


if __name__ == "__main__":
    main()
