# Weight Export Binary Format

## LISTA Weights

```
Offset  Type      Content
------  --------  -------
0       int32     K (number of layers)
4       int32     n (signal dimension)
8       int32     m (measurement dimension)

For each layer k = 0, 1, ..., K-1:
  +0    float64[n×n]  S_k (lateral inhibition matrix, column-major)
  +8n²  float64[n×m]  W_e_k (encoder weight matrix, column-major)
  +8nm  float64[n]    theta_k (threshold vector)
```

**Total size**: 12 + K × (8n² + 8nm + 8n) bytes

**Column-major note**: Eigen stores matrices in column-major order by default.
When exporting from NumPy (row-major), transpose the matrix before writing
so that `Eigen::Map` reads the data correctly.

## ALISTA Weights

```
Offset  Type      Content
------  --------  -------
0       int32     K (number of layers)

For each layer k = 0, 1, ..., K-1:
  +0    float64   gamma_k (step size)
  +8    float64   theta_k (threshold)
```

**Total size**: 4 + K × 16 bytes

**Note**: The sensing matrix A must be provided separately (as a .npy file or
hardcoded). The analytic weight W = (A^T A + α I)^{-1} A^T is computed at
runtime in C++.

## Example Sizes (n=500, m=250, K=16)

- LISTA: 12 + 16 × (2,000,000 + 1,000,000 + 4,000) = ~48.1 MB
- ALISTA: 4 + 16 × 16 = 260 bytes
