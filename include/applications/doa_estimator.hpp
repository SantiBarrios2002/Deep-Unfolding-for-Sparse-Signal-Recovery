#pragma once

#include <Eigen/Dense>
#include <vector>

namespace unfolding {

/// Build the steering matrix D ∈ ℂ^{M × N} for a ULA
/// M = antennas, d_lambda = d/λ, theta_grid in radians
Eigen::MatrixXcd build_steering_matrix(int M, double d_lambda,
                                        const Eigen::VectorXd& theta_grid);

/// DOA problem instance (complex domain)
struct DOAProblem {
    Eigen::MatrixXcd D;              // M × N steering matrix
    Eigen::VectorXcd y;              // M-dim received signal
    Eigen::VectorXd theta_grid;      // N grid angles (radians)
    std::vector<double> true_doas;   // ground-truth DOAs (radians)
    int num_sources;
    double snr_db;
};

/// Generate a DOA problem with random source signals
DOAProblem generate_doa_problem(int M, double d_lambda,
                                 const Eigen::VectorXd& theta_grid,
                                 const std::vector<double>& true_doas_deg,
                                 double snr_db = 20.0,
                                 unsigned int seed = 42);

/// Extract DOA estimates from a recovered sparse signal x̂
/// Returns angles (degrees) of the k largest peaks
std::vector<double> extract_doas(const Eigen::VectorXd& x_hat,
                                  const Eigen::VectorXd& theta_grid_deg,
                                  int num_sources);

}  // namespace unfolding
