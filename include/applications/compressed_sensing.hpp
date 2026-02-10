#pragma once

#include <Eigen/Dense>

namespace unfolding {

/// A synthetic compressed sensing problem instance
struct CSProblem {
    Eigen::MatrixXd A;     // m × n sensing matrix
    Eigen::VectorXd x;     // n-dim ground truth (k-sparse)
    Eigen::VectorXd y;     // m-dim measurements (y = Ax + noise)
    double snr_db;
    int sparsity;
};

/// Generate a random compressed sensing problem
/// A: Gaussian with normalized columns, x: k-sparse N(0,1), y = Ax + noise
CSProblem generate_cs_problem(int m, int n, int k, double snr_db = 40.0,
                               unsigned int seed = 42);

}  // namespace unfolding
