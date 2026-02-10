#pragma once

#include <Eigen/Dense>
#include <random>

namespace unfolding {

/// Compute the Lipschitz constant L = largest eigenvalue of A^T A
inline double compute_lipschitz_constant(const Eigen::MatrixXd& A) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A.transpose() * A);
    return solver.eigenvalues().maxCoeff();
}

/// Normalize columns of A to unit ℓ₂ norm (in-place)
inline void normalize_columns(Eigen::MatrixXd& A) {
    for (int j = 0; j < A.cols(); ++j) {
        double norm = A.col(j).norm();
        if (norm > 0.0) {
            A.col(j) /= norm;
        }
    }
}

/// Generate an m×n Gaussian random matrix with normalized columns
inline Eigen::MatrixXd generate_gaussian_matrix(int m, int n,
                                                 unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(m));

    Eigen::MatrixXd A(m, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A(i, j) = dist(gen);
        }
    }
    normalize_columns(A);
    return A;
}

/// Generate a k-sparse signal of dimension n (nonzeros ~ N(0,1))
inline Eigen::VectorXd generate_sparse_signal(int n, int k,
                                               unsigned int seed = 123) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < k; ++i) {
        x(indices[i]) = normal(gen);
    }
    return x;
}

}  // namespace unfolding
