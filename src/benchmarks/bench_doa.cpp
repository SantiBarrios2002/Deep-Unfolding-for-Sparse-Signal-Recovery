/// Experiment 2: DOA Estimation — RMSE vs SNR (Monte Carlo)

#include <iostream>
#include <vector>
#include <cmath>

#include "applications/doa_estimator.hpp"
#include "math/complex_utils.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "solvers/music_doa.hpp"
#include "evaluation/metrics.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 2: DOA Estimation ===\n";
    std::cout << "M=" << cfg.doa_M << " antennas, K=" << cfg.doa_num_sources
              << " sources, grid=" << cfg.doa_grid_points << " points\n\n";

    // Build angle grid: -90° to 90° in 1° steps
    int N = cfg.doa_grid_points;
    Eigen::VectorXd theta_grid_deg = Eigen::VectorXd::LinSpaced(N, -90.0, 90.0);
    Eigen::VectorXd theta_grid_rad = theta_grid_deg * M_PI / 180.0;

    std::vector<double> true_doas = {-30.0, 10.0, 45.0};

    // SNR sweep
    std::vector<double> snr_values = {-5, 0, 5, 10, 15, 20, 25, 30};

    for (double snr : snr_values) {
        double rmse_sum = 0.0;
        int valid_trials = 0;

        for (int trial = 0; trial < cfg.doa_monte_carlo; ++trial) {
            auto prob = unfolding::generate_doa_problem(
                cfg.doa_M, cfg.doa_d_lambda, theta_grid_rad,
                true_doas, snr, trial);

            // Convert complex DOA problem to real-valued for ISTA
            Eigen::VectorXd y_real = unfolding::complex_to_real_stacked(prob.y);
            Eigen::MatrixXd A_real = unfolding::complex_to_real_stacked(prob.D);

            // Solve with ISTA
            unfolding::IstaSolver ista(0.1, 500, 1e-6);
            Eigen::VectorXd x_hat = ista.solve(y_real, A_real);

            // Extract DOAs (use first N entries = real part magnitudes)
            // For real-stacked: x_hat = [Re(x); Im(x)], combine magnitudes
            Eigen::VectorXd x_mag(N);
            for (int i = 0; i < N; ++i) {
                double re = x_hat(i);
                double im = x_hat(i + N);
                x_mag(i) = std::sqrt(re * re + im * im);
            }

            auto est_doas = unfolding::extract_doas(x_mag, theta_grid_deg,
                                                      cfg.doa_num_sources);
            double r = unfolding::doa_rmse(est_doas, true_doas);
            if (std::isfinite(r)) {
                rmse_sum += r;
                ++valid_trials;
            }
        }

        double avg_rmse = (valid_trials > 0) ? rmse_sum / valid_trials : -1.0;
        std::cout << "SNR=" << snr << " dB: ISTA RMSE = " << avg_rmse << " deg\n";
    }

    // TODO: Add FISTA, LISTA, ALISTA, MUSIC comparisons
    // TODO: Write results to CSV

    std::cout << "\nDone.\n";
    return 0;
}
