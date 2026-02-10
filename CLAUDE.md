# CLAUDE.md — Project: Deep Unfolding for Sparse Signal Recovery (Upgraded LISTA)

## Papers

### Anchor Paper (Survey — frames the project as current research)
**Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing**
V. Monga, Y. Li, Y. C. Eldar
*IEEE Signal Processing Magazine*, vol. 38, no. 2, pp. 18–44, March 2021.

- **IEEE Xplore**: https://ieeexplore.ieee.org/document/9363511
- **arXiv PDF**: https://arxiv.org/pdf/1912.10557
- **DOI**: 10.1109/MSP.2020.3016905
- **Citations**: 800+ (Yonina Eldar = same group as KalmanNet)

### Foundation Paper (The method you implement)
**Learning Fast Approximations of Sparse Coding**
K. Gregor and Y. LeCun
*Proceedings of the 27th International Conference on Machine Learning (ICML)*, pp. 399–406, 2010.

- **PDF**: https://icml.cc/Conferences/2010/papers/449.pdf
- **Citations**: 2,500+ (The seminal paper on algorithm unrolling)

### Modern Extension (Upgrade over vanilla LISTA)
**ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA**
J. Liu, X. Chen, Z. Wang, W. Yin
*International Conference on Learning Representations (ICLR)*, 2019.

- **GitHub**: https://github.com/VITA-Group/ALISTA
- **OpenReview**: https://openreview.net/forum?id=B1lnzn0ctQ
- **Key insight**: Analytically computes the weight matrices (reducing trainable params from O(N²) to O(1) per layer), making LISTA practical for large problems.

### Application Paper (DOA estimation via deep unfolding)
**Deep Learning-Enabled One-Bit DoA Estimation**
Z. Esmaeilbeig, R. Zheng, H. V. Poor, M. Soltanalian
*arXiv preprint*, 2024.

- **arXiv**: https://arxiv.org/abs/2405.09712
- **Relevance**: Uses LISTA to solve DOA as a sparse recovery problem — directly connects your project to array signal processing.

### Additional references
- Borgerding et al., "AMP-Inspired Deep Networks for Sparse Linear Inverse Problems," *IEEE TSP*, 2017
- Chen et al., "Theoretical Linear Convergence of Unfolded ISTA," *NeurIPS*, 2018
- DeepFPC for DOA: https://www.sciencedirect.com/science/article/abs/pii/S0165168420302425
- CADMM-Net for DOA: https://arxiv.org/abs/2502.19076
- Efficient Deep Unfolding for OFDM: https://arxiv.org/abs/2210.06588

### GitHub References
- **LISTA (PyTorch)**: https://github.com/kartheekkumar65/LISTA
- **LISTA-SC (PyTorch)**: https://github.com/faisaljayousi/LISTA-SC
- **ALISTA (TensorFlow)**: https://github.com/VITA-Group/ALISTA
- **Ada-LISTA (PyTorch)**: https://github.com/aaberdam/AdaLISTA
- **Hybrid ISTA (PyTorch)**: https://github.com/AutomatonZZY/Hybrid_ISTA
- **adopty (LISTA/ALISTA/SLISTA comparison)**: https://github.com/tomMoral/adopty

## Course Alignment (ASPTA 2026)

- **Primary**: Block 1 — Estimation Theory
  - §1.2 Maximum Likelihood Estimation: LASSO = MAP estimation under a Laplacian prior on x and Gaussian noise → connects directly to ML/MAP framework
  - §1.3 Bayesian Estimation: The sparse prior p(x) ∝ exp(−λ||x||₁) is a Bayesian model; ISTA is coordinate-wise MAP inference
  - §1.1 Sufficient Statistics: The sensing matrix A acts as a compression of the signal; LISTA learns a sufficient statistic for sparse recovery
- **Secondary**: Block 2 — Adaptive Filtering
  - §2.1 RLS / LMS: ISTA is an adaptive gradient descent algorithm. LISTA learns optimal step sizes per layer, analogous to learning the step size in LMS/RLS.
  - §2.2 Kalman Filter: The predict-update structure of the KF is a special case of proximal splitting (predict = prior step, update = likelihood step). ISTA is the same structure for sparse priors.
- **Tertiary**: Block 3 — Detection Theory
  - §3.1 Detection: The soft-thresholding operator in ISTA is a shrinkage estimator that also performs implicit detection (decides which components are zero vs nonzero). This connects to Neyman-Pearson hypothesis testing on each coefficient.
- **Application tie-in**: DOA estimation (Block 1 estimation + array processing) and OFDM channel estimation (Block 2 adaptive filtering) are canonical signal processing problems.
- **ML integration**: Core — the entire project IS the bridge between optimization and deep learning.

### Why Monga 2021 as anchor (not Gregor 2010)

Gregor & LeCun 2010 is 16 years old. Presenting it alone risks feeling dated. Monga et al. 2021 is a comprehensive IEEE Signal Processing Magazine survey (the field's flagship tutorial venue) co-authored by Yonina Eldar (same group as KalmanNet). It frames deep unfolding as a *current* research paradigm with applications across communications, imaging, radar, and speech. Your project implements the foundational method (LISTA) and its modern upgrade (ALISTA), framed within this contemporary survey.

## Project Objective

Implement a C++ sparse signal recovery engine that demonstrates the deep unfolding paradigm:
1. Classical solvers (ISTA, FISTA) as baselines
2. LISTA (Gregor & LeCun 2010) — learned unfolded network
3. ALISTA (Liu et al., ICLR 2019) — analytically initialized LISTA with fewer parameters

Apply all three to:
- **Synthetic compressed sensing** (reproduce the original paper's results)
- **Direction of Arrival (DOA) estimation** (sparse recovery in angular domain — a real SP problem)
- **Audio inpainting** (perceptual demo for the presentation)

Demonstrate that 10–16 learned layers achieve the same accuracy as 500+ classical iterations, with 100× wall-clock speedup in C++.

## Algorithm Summary

### The Problem: Sparse Recovery (LASSO)

Given measurement y ∈ ℝ^m and sensing matrix A ∈ ℝ^{m×n} (m < n, underdetermined), recover sparse x* ∈ ℝ^n:

```
minimize  (1/2)||y − Ax||₂² + λ||x||₁
```

**Bayesian interpretation** (connects to Block 1, §1.3):
- Likelihood: p(y|x) = N(Ax, σ²I) → log-likelihood = −(1/2σ²)||y − Ax||₂²
- Prior: p(x) ∝ exp(−λ||x||₁) (Laplacian / double-exponential prior)
- MAP estimate: x_MAP = argmin [−log p(y|x) − log p(x)] = argmin [(1/2)||y − Ax||₂² + λ||x||₁]

This is exactly the LASSO. The sparse prior enforces that most entries of x are zero.

### 1. ISTA (Iterative Shrinkage-Thresholding Algorithm)

```
Initialize: x₀ = 0

For k = 0, 1, 2, ... until convergence:
    z_k   = x_k − (1/L) · Aᵀ(Ax_k − y)          # gradient step on ||y − Ax||²
    x_{k+1} = soft_threshold(z_k, λ/L)             # proximal step for λ||x||₁

where:
    soft_threshold(u, α) = sign(u) · max(|u| − α, 0)    (element-wise)
    L = ||AᵀA||₂ = largest eigenvalue of AᵀA              (Lipschitz constant)
```

Convergence rate: O(1/k). Typically requires 100–500 iterations for high accuracy.

### 2. FISTA (Fast ISTA, Nesterov acceleration)

```
Initialize: x₀ = 0, t₀ = 1

For k = 0, 1, 2, ...:
    z_k     = x_k − (1/L) · Aᵀ(Ax_k − y)
    x_{k+1} = soft_threshold(z_k, λ/L)
    t_{k+1} = (1 + √(1 + 4t_k²)) / 2
    x_{k+1} = x_{k+1} + ((t_k − 1) / t_{k+1}) · (x_{k+1} − x_k)    # momentum
```

Convergence rate: O(1/k²). ~3-5× faster than ISTA.

### 3. LISTA (Learned ISTA — Gregor & LeCun 2010)

Unroll ISTA for exactly K layers. Untie parameters across layers:

```
Layer 0: c₀ = soft_threshold(W_e · y, θ₀)

For layer k = 1 to K−1:
    c_k = soft_threshold(S_k · c_{k-1} + W_e_k · y, θ_k)

Output: x̂ = c_K
```

**Learned parameters per layer**: S_k ∈ ℝ^{n×n}, W_e_k ∈ ℝ^{n×m}, θ_k ∈ ℝ^n
**Total parameters**: K × (n² + nm + n)

**Connection to ISTA**: If we set S_k = I − (1/L)AᵀA and W_e_k = (1/L)Aᵀ for all k,
we recover ISTA exactly. LISTA *learns* better matrices from data.

**Training**: Supervised. Generate (y, x*) pairs. Minimize ||c_K − x*||₂² via backpropagation through all K layers.

### 4. ALISTA (Analytic LISTA — Liu et al., ICLR 2019)

**Key insight**: The weight matrices W_e, S in LISTA can be computed analytically from A (no learning needed). Only the thresholds θ_k and a scalar step size γ_k per layer need to be learned.

```
Precompute (no training):
    W = (AᵀA + αI)⁻¹Aᵀ                    # analytic encoder weight
    S = I − W·A                              # analytic lateral inhibition

Layer k:
    c_k = soft_threshold(c_{k-1} − γ_k · W · (A·c_{k-1} − y), θ_k)

Learned parameters per layer: γ_k ∈ ℝ (scalar), θ_k ∈ ℝ (scalar or per-entry)
Total parameters: K × 2    (vs LISTA's K × (n² + nm + n))
```

**Why this matters**: For DOA estimation with n = 360 (1° grid), LISTA needs 360² = 130,000 parameters per layer. ALISTA needs 2 per layer. This makes DOA-scale problems tractable.

## Experiments

### Experiment 1: Synthetic Compressed Sensing (Reproduce Gregor & LeCun Fig. 2)

```
Setup:
    n = 500 (signal dimension)
    m = 250 (measurement dimension)
    A: random Gaussian, columns normalized to unit ℓ₂ norm
    x*: k-sparse, k = 25, nonzero entries ~ N(0, 1)
    y = Ax* + noise (SNR = 40 dB)
    K = 16 layers for LISTA/ALISTA

Metric: NMSE (dB) = 10·log₁₀(||x̂ − x*||² / ||x*||²) vs layers/iterations

Expected result:
    ISTA at 500 iterations ≈ −40 dB NMSE
    FISTA at 150 iterations ≈ −40 dB NMSE
    LISTA at 16 layers ≈ −40 dB NMSE
    ALISTA at 16 layers ≈ −40 dB NMSE (matching LISTA with 99.99% fewer parameters)
```

### Experiment 2: DOA Estimation (The Signal Processing Application)

**Problem**: Estimate directions of arrival of K narrow-band sources from an M-element ULA.

```
Signal model:
    y = A(θ) · s + n ∈ ℂ^M

where:
    A(θ) = [a(θ₁), ..., a(θ_K)]           # steering matrix
    a(θ) = [1, e^{j2πd sin(θ)/λ}, ..., e^{j2π(M-1)d sin(θ)/λ}]ᵀ   # steering vector
    s ∈ ℂ^K = source signals
    n ~ CN(0, σ²I) = noise

Sparse formulation:
    Discretize θ onto a grid: θ_grid = {−90°, −89°, ..., 89°, 90°} (N = 181 bins)
    Build dictionary: D = [a(θ_grid₁), ..., a(θ_grid_N)] ∈ ℂ^{M×N}
    DOA estimation ≡ sparse recovery: y = D·x + n, where x has K nonzero entries
    The positions of nonzero entries in x indicate the source directions

Setup:
    M = 16 antennas, d = λ/2 (half-wavelength spacing)
    K = 3 sources at θ = {−30°, 10°, 45°}
    N = 181 grid points (1° resolution)
    SNR sweep: −5 to 30 dB
    100 Monte Carlo trials per SNR

Baselines: MUSIC, ISTA (500 iter), FISTA (200 iter)
Proposed: LISTA (16 layers), ALISTA (16 layers)

Metrics:
    - RMSE of estimated DOAs vs SNR
    - Detection probability (DOA within ±1° of true) vs SNR
    - Spatial spectrum plots (angular power as a function of θ)

Expected results:
    - LISTA/ALISTA match ISTA's accuracy at 16 layers
    - At low SNR (< 5 dB), learned methods can outperform classical by 2–3 dB
    - Wall-clock time: LISTA ~10× faster than FISTA for same accuracy
```

**Presentation value**: Plot the angular spectrum — ISTA (500 iter) vs LISTA (16 layers).
Both show the same three sharp peaks at −30°, 10°, 45°. But LISTA ran in microseconds.

### Experiment 3: Speed-Accuracy Benchmark (The Wow Factor)

```
Measure wall-clock time in C++ (std::chrono::high_resolution_clock):

Method          | Iterations/Layers | NMSE (dB) | Time (μs) | Speedup
----------------|-------------------|-----------|-----------|--------
ISTA            | 500               | −40       | ~2000     | 1×
FISTA           | 150               | −40       | ~700      | 3×
LISTA           | 16                | −40       | ~30       | 67×
ALISTA          | 16                | −40       | ~20       | 100×

Plot: NMSE vs wall-clock time (log scale). LISTA/ALISTA curve dominates.
```

### Experiment 4: Audio Inpainting (Perceptual Demo)

```
Setup:
    Audio: 3-second clip, 16 kHz sampling rate
    Degradation: Randomly delete 50% of samples (simulating packet loss)
    Dictionary: Windowed DCT basis (audio is approximately sparse in DCT domain)
    A = sampling operator (binary mask) × DCT

Recovery: Apply ISTA (500 iter) and LISTA (16 layers) to each overlapping frame

Output:
    - Original .wav
    - Corrupted .wav (gaps → crackling)
    - ISTA-recovered .wav
    - LISTA-recovered .wav

Demo: Play all four during presentation.
```

### Experiment 5: Generalization & Ablation

```
a) Sparsity generalization:
   Train on k = 25. Test on k = {10, 15, 20, 25, 30, 40, 50}.
   Does LISTA still work? (Usually yes, within a range.)

b) Layer count sweep:
   K = {2, 4, 8, 12, 16, 20, 24} layers. Plot NMSE vs K.
   Show diminishing returns past ~16 layers.

c) LISTA vs ALISTA parameter count:
   Bar chart: total trainable parameters for n = 500.
   LISTA: 16 × (500² + 500×250 + 500) = ~6 million
   ALISTA: 16 × 2 = 32. That's a 200,000× reduction.

d) SNR robustness:
   Train at SNR = 40 dB. Test at SNR = {10, 20, 30, 40, ∞}.
   ALISTA degrades more gracefully than LISTA (analytic initialization acts as regularizer).
```

## Project Structure

```
deep-unfolding-cpp/
├── CMakeLists.txt
├── README.md
├── CLAUDE.md                              # This file
│
├── include/
│   ├── math/
│   │   ├── soft_threshold.hpp             # Soft-thresholding operator (element-wise)
│   │   ├── matrix_utils.hpp               # Lipschitz constant, column normalization
│   │   └── complex_utils.hpp              # Complex-valued operations for DOA
│   │
│   ├── solvers/
│   │   ├── solver_base.hpp                # Abstract interface: solve(y, A) → x̂
│   │   ├── ista_solver.hpp                # Classical ISTA
│   │   ├── fista_solver.hpp               # Classical FISTA
│   │   └── music_doa.hpp                  # MUSIC algorithm (DOA baseline)
│   │
│   ├── neural/
│   │   ├── lista_network.hpp              # LISTA forward pass: K layers of S·c + W·y + thresh
│   │   ├── alista_network.hpp             # ALISTA: analytic W, S + learned γ, θ
│   │   ├── weights_loader.hpp             # Load trained weights from binary file
│   │   └── layer.hpp                      # Single unfolded layer abstraction
│   │
│   ├── applications/
│   │   ├── compressed_sensing.hpp         # Synthetic CS problem generator
│   │   ├── doa_estimator.hpp              # DOA: steering matrix, grid, peak extraction
│   │   └── audio_inpainter.hpp            # Audio frame processing with DCT dictionary
│   │
│   ├── evaluation/
│   │   ├── metrics.hpp                    # NMSE, RMSE, detection probability
│   │   └── timer.hpp                      # std::chrono wrapper for benchmarking
│   │
│   └── utils/
│       ├── csv_writer.hpp
│       ├── wav_io.hpp                     # Read/write .wav files (or use AudioFile header)
│       └── config.hpp
│
├── src/
│   ├── solvers/
│   │   ├── ista_solver.cpp                # ~40 lines
│   │   ├── fista_solver.cpp               # ~50 lines
│   │   └── music_doa.cpp                  # ~80 lines (eigendecomposition + spectrum)
│   │
│   ├── neural/
│   │   ├── lista_network.cpp              # ~60 lines (matrix multiply + threshold, K times)
│   │   ├── alista_network.cpp             # ~50 lines (even simpler — analytic matrices)
│   │   └── weights_loader.cpp             # ~40 lines (binary file → Eigen matrices)
│   │
│   ├── applications/
│   │   ├── compressed_sensing.cpp         # ~60 lines (generate A, x*, y)
│   │   ├── doa_estimator.cpp              # ~120 lines (steering matrix, grid search, peaks)
│   │   └── audio_inpainter.cpp            # ~100 lines (frame-wise DCT + recovery)
│   │
│   ├── benchmarks/
│   │   ├── bench_synthetic.cpp            # Experiment 1: NMSE vs iterations/layers
│   │   ├── bench_doa.cpp                  # Experiment 2: DOA RMSE vs SNR
│   │   ├── bench_speed.cpp                # Experiment 3: wall-clock timing
│   │   ├── bench_audio.cpp                # Experiment 4: audio inpainting
│   │   └── bench_ablation.cpp             # Experiment 5: generalization tests
│   │
│   └── main.cpp                           # CLI entry point
│
├── training/                              # Python (PyTorch)
│   ├── requirements.txt                   # torch, numpy, matplotlib, soundfile
│   ├── train_lista.py                     # LISTA training loop
│   ├── train_alista.py                    # ALISTA training (much faster — fewer params)
│   ├── data_generator.py                  # Synthetic (A, x*, y) pair generator
│   ├── doa_data_generator.py              # DOA-specific data: steering matrices + sources
│   ├── export_weights.py                  # Export W, S, θ, γ to binary for C++
│   ├── export_weights_format.md           # Document the binary format
│   └── visualization.py                   # Loss curves, spectrum plots
│
├── scripts/
│   ├── plot_nmse_vs_iterations.py         # Experiment 1 figure
│   ├── plot_doa_spectrum.py               # Experiment 2: angular spectrum
│   ├── plot_doa_rmse_vs_snr.py            # Experiment 2: RMSE vs SNR
│   ├── plot_speed_accuracy.py             # Experiment 3: accuracy vs latency
│   ├── plot_audio_waveforms.py            # Experiment 4: original vs corrupted vs recovered
│   ├── plot_ablation.py                   # Experiment 5 figures
│   └── generate_presentation_figs.py      # All figures for slides
│
├── data/
│   ├── audio/                             # Sample .wav files
│   ├── weights/                           # Exported model parameters (binary)
│   └── results/                           # Benchmark output CSVs
│
├── tests/
│   ├── test_soft_threshold.cpp            # Verify: soft_threshold(0.5, 0.3) = 0.2
│   ├── test_ista_convergence.cpp          # ISTA converges to known solution
│   ├── test_lista_matches_ista.cpp        # LISTA with ISTA weights = ISTA output
│   ├── test_steering_vector.cpp           # Steering vector phase correctness
│   └── test_weights_loader.cpp            # Round-trip: Python export → C++ load
│
└── external/
    ├── eigen/                             # Eigen 3.4+ (the only required dependency)
    └── AudioFile/                         # Optional: header-only .wav reader
```

## Dependencies

### C++ (everything)
- **Eigen 3.4+**: Dense matrix operations (MatrixXd, VectorXd, complex support)
- **C++17**: std::filesystem, std::chrono, structured bindings
- **CMake 3.18+**: Build system
- **AudioFile** (optional): Header-only .wav I/O (https://github.com/adamstark/AudioFile)
- **No LibTorch / No ONNX Runtime**: LISTA is just MatMul + soft-threshold. Pure Eigen.

### Python (training only)
- **PyTorch >= 1.12**: Training + autograd for backpropagation through unrolled layers
- **NumPy**: Data generation
- **matplotlib**: Visualization
- **soundfile** (optional): Audio file I/O

## Key Implementation Notes

### 1. LISTA Forward Pass in Pure Eigen (the showcase)
```cpp
// This is the ENTIRE inference engine. No ML framework needed.
Eigen::VectorXd lista_forward(const Eigen::VectorXd& y,
                               const std::vector<Eigen::MatrixXd>& S,    // K matrices n×n
                               const std::vector<Eigen::MatrixXd>& W_e,  // K matrices n×m
                               const std::vector<Eigen::VectorXd>& theta, // K threshold vectors
                               int K) {
    Eigen::VectorXd c = soft_threshold(W_e[0] * y, theta[0]);
    for (int k = 1; k < K; ++k) {
        c = soft_threshold(S[k] * c + W_e[k] * y, theta[k]);
    }
    return c;
}

// Soft-thresholding: the proximal operator for ℓ₁ norm
Eigen::VectorXd soft_threshold(const Eigen::VectorXd& u, const Eigen::VectorXd& alpha) {
    return u.array().sign() * (u.array().abs() - alpha.array()).max(0.0);
}
```

This is ~15 lines. The entire C++ inference engine for a learned sparse coder. That's the point.

### 2. ALISTA Forward Pass (Even Simpler)
```cpp
Eigen::VectorXd alista_forward(const Eigen::VectorXd& y,
                                const Eigen::MatrixXd& W,     // analytic, precomputed ONCE
                                const Eigen::MatrixXd& A,     // sensing matrix
                                const std::vector<double>& gamma,  // K learned step sizes
                                const std::vector<double>& theta,  // K learned thresholds
                                int K) {
    Eigen::VectorXd c = Eigen::VectorXd::Zero(W.cols());
    for (int k = 0; k < K; ++k) {
        Eigen::VectorXd residual = A * c - y;
        c = soft_threshold(c - gamma[k] * (W * residual), theta[k]);
    }
    return c;
}
```

Total learned parameters: K × 2 scalars. Everything else is computed from A.

### 3. Steering Matrix for DOA
```cpp
Eigen::MatrixXcd build_steering_matrix(int M, double d_lambda,
                                        const Eigen::VectorXd& theta_grid) {
    // M antennas, d_lambda = d/λ (typically 0.5), theta_grid in radians
    int N = theta_grid.size();
    Eigen::MatrixXcd D(M, N);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            double phase = 2.0 * M_PI * i * d_lambda * std::sin(theta_grid(j));
            D(i, j) = std::complex<double>(std::cos(phase), std::sin(phase));
        }
    }
    return D;
}
```

### 4. Complex-Valued LISTA for DOA
DOA involves complex signals. Two options:
- **Option A (recommended)**: Stack real and imaginary parts → real-valued LISTA on 2M-dimensional problem
  ```
  y_real = [Re(y); Im(y)] ∈ ℝ^{2M}
  A_real = [Re(D), -Im(D); Im(D), Re(D)] ∈ ℝ^{2M × 2N}
  ```
- **Option B**: Implement complex LISTA directly using Eigen's `MatrixXcd`

Option A is simpler and lets you reuse the same real-valued LISTA code for both CS and DOA.

### 5. Weight Export Format
```python
# Python: export trained weights to binary
import struct
import numpy as np

def export_weights(filepath, S_list, W_e_list, theta_list):
    with open(filepath, 'wb') as f:
        K = len(S_list)
        n, m = W_e_list[0].shape
        f.write(struct.pack('iii', K, n, m))
        for k in range(K):
            f.write(S_list[k].astype(np.float64).tobytes())      # n×n
            f.write(W_e_list[k].astype(np.float64).tobytes())    # n×m
            f.write(theta_list[k].astype(np.float64).tobytes())  # n
```

```cpp
// C++: load weights from binary
void load_weights(const std::string& path,
                  std::vector<Eigen::MatrixXd>& S,
                  std::vector<Eigen::MatrixXd>& W_e,
                  std::vector<Eigen::VectorXd>& theta) {
    std::ifstream f(path, std::ios::binary);
    int K, n, m;
    f.read(reinterpret_cast<char*>(&K), sizeof(int));
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    f.read(reinterpret_cast<char*>(&m), sizeof(int));
    
    S.resize(K); W_e.resize(K); theta.resize(K);
    for (int k = 0; k < K; ++k) {
        S[k].resize(n, n);
        f.read(reinterpret_cast<char*>(S[k].data()), n * n * sizeof(double));
        W_e[k].resize(n, m);
        f.read(reinterpret_cast<char*>(W_e[k].data()), n * m * sizeof(double));
        theta[k].resize(n);
        f.read(reinterpret_cast<char*>(theta[k].data()), n * sizeof(double));
    }
}
```

### 6. ISTA as Adaptive Filter (Block 2 tie-in)
Frame ISTA as a special case of gradient descent with projection, connecting to LMS/RLS:

```
LMS:  w_{k+1} = w_k + μ · x_k · (d_k − x_kᵀ w_k)           # adaptive filter
ISTA: x_{k+1} = S_λ(x_k + (1/L) · Aᵀ(y − A x_k))           # sparse estimator

Both: θ_{k+1} = θ_k + step · gradient(data_fidelity)          # same structure!
```

The difference: ISTA adds the soft-thresholding projection step (proximal operator for ℓ₁).
LISTA learns the optimal step size per layer, just as optimal step-size selection is key in LMS/RLS.

### 7. Performance: C++ vs Python
For n = 500, m = 250, K = 16 layers:
- **Python (PyTorch, GPU)**: ~0.5 ms per signal (batched), includes overhead
- **Python (NumPy, CPU)**: ~2 ms per signal
- **C++ (Eigen, CPU)**: ~20 μs per signal (100× faster than NumPy)
- **C++ (Eigen + BLAS)**: ~10 μs per signal

For DOA with n = 181, m = 16, K = 16 layers:
- C++ LISTA inference: ~2 μs per snapshot (500,000 DOA estimates per second)
- This is fast enough for real-time radar/sonar processing

## Implementation Timeline (6 weeks)

| Week | Task |
|------|------|
| 1 | **Math core**: Soft-thresholding, ISTA, FISTA in C++. Validate on synthetic CS (n=500, m=250). Reproduce ISTA convergence curve. |
| 2 | **Training**: PyTorch LISTA and ALISTA training scripts. Reproduce Gregor & LeCun Fig. 2 (NMSE vs layers). Export weights to binary. |
| 3 | **C++ inference**: Implement LISTA and ALISTA forward pass in Eigen. Load weights. Verify output matches Python to float precision. Run speed benchmark. |
| 4 | **DOA application**: Build steering matrix. Implement MUSIC baseline. Train LISTA/ALISTA on DOA data. Run DOA experiments (RMSE vs SNR). |
| 5 | **Audio demo**: Implement DCT-based audio inpainting. Generate .wav files for presentation. Run generalization/ablation experiments. |
| 6 | **Polish**: Generate all figures. Prepare slides. Clean repo. Buffer for debugging. |

## Presentation Strategy

### 5-min Preview (21 May)
- **Slide 1**: "Iterative algorithms are precise but slow. Neural networks are fast but uninterpretable. Algorithm unrolling gives you both." Show Monga et al.'s Fig. 1 (iteration → network layer diagram).
- **Slide 2**: The LASSO problem → Bayesian MAP estimation → ISTA as coordinate-wise inference.
- **Slide 3**: LISTA: unroll the loop, untie the weights, train end-to-end.
- **Slide 4**: My plan: ISTA → LISTA → ALISTA, applied to DOA estimation and audio recovery.

### Full Presentation (June)
- **The Race** (Experiment 3): Live C++ terminal output showing timing comparison
- **DOA Spectrum** (Experiment 2): Side-by-side angular spectra: ISTA (500 iter) vs LISTA (16 layers) — identical peaks, 100× faster
- **Audio Demo** (Experiment 4): Play corrupted → recovered audio
- **The Punchline**: ALISTA achieves the same accuracy as LISTA with 200,000× fewer parameters. This is what "interpretable ML" means — the math tells you the right architecture.
- **Theory connection**: Walk through how ISTA = MAP inference under Laplacian prior, connecting back to Block 1's Bayesian estimation framework.

## Why This Upgraded Version Is Stronger

| Aspect | Original LISTA Proposal | Upgraded Version |
|--------|:-----------------------:|:----------------:|
| Anchor paper | Gregor & LeCun 2010 (16 years old) | Monga et al. 2021 IEEE SPM (current) |
| Method | LISTA only | ISTA → FISTA → LISTA → ALISTA progression |
| SP application | Generic CS + audio | DOA estimation (array processing) + audio |
| Course alignment | Block 1 (tangential) | Block 1 (MAP/Bayesian) + Block 2 (adaptive) + Block 3 (detection) |
| Parameter discussion | None | ALISTA's 200,000× reduction is a key result |
| Baselines | ISTA only | ISTA + FISTA + MUSIC (for DOA) |
| Theoretical depth | Proximal operators | Bayesian MAP + proximal + Lipschitz theory + analytic weight initialization |
| Reference code | Sparse | 6+ GitHub repos across LISTA/ALISTA variants |

## References

- K. Gregor, Y. LeCun, "Learning Fast Approximations of Sparse Coding," ICML 2010
- V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling," IEEE SPM 2021
- J. Liu, X. Chen, Z. Wang, W. Yin, "ALISTA," ICLR 2019
- X. Chen et al., "Theoretical Linear Convergence of Unfolded ISTA," NeurIPS 2018
- I. Daubechies, M. Defrise, C. De Mol, "An iterative thresholding algorithm for linear inverse problems," Comm. Pure Appl. Math., 2004 (original ISTA)
- A. Beck, M. Teboulle, "A fast iterative shrinkage-thresholding algorithm," SIAM J. Imaging Sci., 2009 (FISTA)
- S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*, 1993 (course textbook)
- R. Tibshirani, "Regression Shrinkage and Selection via the LASSO," JRSS-B, 1996
