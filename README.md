# Deep Unfolding for Sparse Signal Recovery

A C++ implementation of algorithm unrolling for sparse signal recovery, demonstrating the bridge between classical optimization and deep learning.

This project implements **LISTA** (Learned ISTA) and **ALISTA** (Analytic LISTA) for fast sparse signal recovery, with applications to compressed sensing, Direction of Arrival (DOA) estimation, and audio inpainting.

## Overview

**Algorithm unrolling** transforms iterative optimization algorithms into trainable neural networks by:
1. Unrolling a fixed number of iterations into network layers
2. Replacing fixed parameters with learnable ones
3. Training end-to-end via backpropagation

This project demonstrates that 16 learned layers can match 500+ classical iterations, achieving **100x speedup** with equivalent accuracy.

### Key Features

- **Classical solvers**: ISTA, FISTA (Nesterov-accelerated)
- **Learned solvers**: LISTA, ALISTA (200,000x fewer parameters than LISTA)
- **Applications**: Compressed sensing, DOA estimation, audio inpainting
- **Pure C++ inference**: No ML framework dependencies (just Eigen)
- **PyTorch training**: Scripts for training LISTA/ALISTA networks

## The Problem: Sparse Recovery (LASSO)

Given measurements `y` and sensing matrix `A`, recover sparse signal `x*`:

```
minimize  (1/2)||y - Ax||_2^2 + lambda||x||_1
```

This is equivalent to MAP estimation under a Laplacian prior on `x`.

## Algorithms

| Method | Convergence | Typical Iterations | Parameters per Layer |
|--------|-------------|-------------------|---------------------|
| ISTA   | O(1/k)      | 500               | 0 (fixed)           |
| FISTA  | O(1/k^2)    | 150               | 0 (fixed)           |
| LISTA  | -           | 16 layers         | n^2 + nm + n        |
| ALISTA | -           | 16 layers         | 2 (scalar)          |

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── math/              # Soft-thresholding, matrix utilities
│   ├── solvers/           # ISTA, FISTA, MUSIC
│   ├── neural/            # LISTA, ALISTA networks
│   ├── applications/      # CS, DOA, audio inpainting
│   ├── evaluation/        # Metrics (NMSE, RMSE), timing
│   └── utils/             # CSV, WAV I/O, config
├── src/
│   ├── solvers/           # Solver implementations
│   ├── neural/            # Network implementations
│   ├── applications/      # Application implementations
│   ├── benchmarks/        # Experiment executables
│   └── main.cpp           # CLI entry point
├── training/              # PyTorch training scripts
│   ├── train_lista.py     # LISTA training
│   ├── train_alista.py    # ALISTA training
│   ├── data_generator.py  # Synthetic data generation
│   └── export_weights.py  # Export to C++ binary format
├── tests/                 # Unit tests
└── data/                  # Weights, audio, results
```

## Requirements

### C++ (Inference)

- **CMake** >= 3.18
- **C++17** compatible compiler
- **Eigen 3.4+** (auto-fetched if not found)

### Python (Training)

- Python >= 3.8
- PyTorch >= 1.12
- NumPy, Matplotlib, SciPy
- soundfile (for audio)

## Building

```bash
# Configure
mkdir build && cd build
cmake ..

# Build all targets
cmake --build .

# Run tests
ctest --output-on-failure
```

### Build Options

```bash
# Enable AudioFile for .wav I/O
cmake -DUSE_AUDIOFILE=ON ..
```

## Usage

### Training (Python)

```bash
cd training

# Install dependencies
pip install -r requirements.txt

# Train LISTA network
python train_lista.py --layers 16 --epochs 100

# Train ALISTA network (faster - fewer parameters)
python train_alista.py --layers 16 --epochs 100

# Export weights for C++ inference
python export_weights.py --model lista --output ../data/weights/lista_16.bin
```

### Inference (C++)

```bash
# Run main executable
./deep_unfolding

# Run benchmarks
./bench_synthetic   # Experiment 1: NMSE vs iterations/layers
./bench_doa         # Experiment 2: DOA RMSE vs SNR
./bench_speed       # Experiment 3: Wall-clock timing
./bench_audio       # Experiment 4: Audio inpainting
./bench_ablation    # Experiment 5: Generalization tests
```

## Experiments

### 1. Synthetic Compressed Sensing

Reproduce Gregor & LeCun (ICML 2010) Figure 2.

- n = 500 (signal dimension), m = 250 (measurements)
- k = 25 sparse entries
- Compare ISTA (500 iter), FISTA (150 iter), LISTA (16 layers), ALISTA (16 layers)

### 2. DOA Estimation

Direction of Arrival estimation using sparse recovery on a steering matrix.

- M = 16 antenna ULA, N = 181 grid points (1 degree resolution)
- Compare against MUSIC baseline
- Evaluate RMSE vs SNR

### 3. Speed-Accuracy Benchmark

Wall-clock timing comparison:

| Method | Iterations/Layers | NMSE (dB) | Time (us) | Speedup |
|--------|-------------------|-----------|-----------|---------|
| ISTA   | 500               | -40       | ~2000     | 1x      |
| FISTA  | 150               | -40       | ~700      | 3x      |
| LISTA  | 16                | -40       | ~30       | 67x     |
| ALISTA | 16                | -40       | ~20       | 100x    |

### 4. Audio Inpainting

Recover audio with 50% missing samples using DCT dictionary.

### 5. Ablation Studies

- Sparsity generalization (train k=25, test k=10-50)
- Layer count sweep (K = 2, 4, 8, 12, 16, 20, 24)
- SNR robustness

## References

### Foundation Paper
> K. Gregor and Y. LeCun, "Learning Fast Approximations of Sparse Coding," *ICML*, 2010.

### Modern Extension
> J. Liu, X. Chen, Z. Wang, W. Yin, "ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA," *ICLR*, 2019.

### Survey (Anchor Paper)
> V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing," *IEEE Signal Processing Magazine*, vol. 38, no. 2, pp. 18-44, March 2021.

### Classical Algorithms
- I. Daubechies, M. Defrise, C. De Mol, "An iterative thresholding algorithm for linear inverse problems," *Comm. Pure Appl. Math.*, 2004.
- A. Beck, M. Teboulle, "A fast iterative shrinkage-thresholding algorithm," *SIAM J. Imaging Sci.*, 2009.

## License

MIT License

## Acknowledgments

This project was developed for the Advanced Signal Processing: Theory and Applications (ASPTA) course at Universitat Politecnica de Catalunya (UPC), 2026.
