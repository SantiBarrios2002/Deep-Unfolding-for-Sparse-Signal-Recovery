#include "applications/doa_estimator.hpp"
#include <cmath>
#include <complex>
#include <random>
#include <algorithm>
#include <numeric>

namespace unfolding {

Eigen::MatrixXcd build_steering_matrix(int M, double d_lambda,
                                         const Eigen::VectorXd& theta_grid) {
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

DOAProblem generate_doa_problem(int M, double d_lambda,
                                  const Eigen::VectorXd& theta_grid,
                                  const std::vector<double>& true_doas_deg,
                                  double snr_db, unsigned int seed) {
    DOAProblem prob;
    prob.D = build_steering_matrix(M, d_lambda, theta_grid);
    prob.theta_grid = theta_grid;
    prob.true_doas = true_doas_deg;
    prob.num_sources = static_cast<int>(true_doas_deg.size());
    prob.snr_db = snr_db;

    int K = prob.num_sources;
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Build steering matrix for true DOAs
    Eigen::VectorXd true_doas_rad(K);
    for (int i = 0; i < K; ++i) {
        true_doas_rad(i) = true_doas_deg[i] * M_PI / 180.0;
    }
    Eigen::MatrixXcd A_true = build_steering_matrix(M, d_lambda, true_doas_rad);

    // Random source signals (complex)
    Eigen::VectorXcd s(K);
    for (int i = 0; i < K; ++i) {
        s(i) = std::complex<double>(dist(gen), dist(gen)) / std::sqrt(2.0);
    }

    // Clean signal
    Eigen::VectorXcd y_clean = A_true * s;

    // Add complex Gaussian noise
    Eigen::VectorXcd noise(M);
    for (int i = 0; i < M; ++i) {
        noise(i) = std::complex<double>(dist(gen), dist(gen)) / std::sqrt(2.0);
    }

    double signal_power = y_clean.squaredNorm() / M;
    double noise_power = signal_power * std::pow(10.0, -snr_db / 10.0);
    double noise_scale = std::sqrt(noise_power / (noise.squaredNorm() / M));
    noise *= noise_scale;

    prob.y = y_clean + noise;

    return prob;
}

std::vector<double> extract_doas(const Eigen::VectorXd& x_hat,
                                   const Eigen::VectorXd& theta_grid_deg,
                                   int num_sources) {
    // Find the k largest magnitude entries
    int N = x_hat.size();
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + num_sources, indices.end(),
                      [&](int a, int b) {
                          return std::abs(x_hat(a)) > std::abs(x_hat(b));
                      });

    std::vector<double> doas;
    for (int i = 0; i < num_sources && i < N; ++i) {
        doas.push_back(theta_grid_deg(indices[i]));
    }

    return doas;
}

}  // namespace unfolding
