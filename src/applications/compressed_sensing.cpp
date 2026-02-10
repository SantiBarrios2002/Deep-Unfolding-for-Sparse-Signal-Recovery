#include "applications/compressed_sensing.hpp"
#include "math/matrix_utils.hpp"
#include <random>
#include <cmath>

namespace unfolding {

CSProblem generate_cs_problem(int m, int n, int k, double snr_db,
                                unsigned int seed) {
    CSProblem prob;
    prob.snr_db = snr_db;
    prob.sparsity = k;

    // Generate sensing matrix and sparse signal
    prob.A = generate_gaussian_matrix(m, n, seed);
    prob.x = generate_sparse_signal(n, k, seed + 1);

    // Clean measurements
    Eigen::VectorXd y_clean = prob.A * prob.x;

    // Add noise at specified SNR
    std::mt19937 gen(seed + 2);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::VectorXd noise(m);
    for (int i = 0; i < m; ++i) noise(i) = dist(gen);

    double signal_power = y_clean.squaredNorm() / m;
    double noise_power = signal_power * std::pow(10.0, -snr_db / 10.0);
    noise *= std::sqrt(noise_power / (noise.squaredNorm() / m));

    prob.y = y_clean + noise;

    return prob;
}

}  // namespace unfolding
