/// Experiment 2: DOA Estimation — RMSE vs SNR (Monte Carlo)

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

#include "applications/doa_estimator.hpp"
#include "math/complex_utils.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "solvers/music_doa.hpp"
#include "evaluation/metrics.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

/// Helper: extract DOA magnitudes from real-stacked solution
Eigen::VectorXd extract_magnitudes(const Eigen::VectorXd& x_hat, int N) {
    Eigen::VectorXd x_mag(N);
    for (int i = 0; i < N; ++i) {
        double re = x_hat(i);
        double im = x_hat(i + N);
        x_mag(i) = std::sqrt(re * re + im * im);
    }
    return x_mag;
}

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 2: DOA Estimation ===\n";
    std::cout << "M=" << cfg.doa_M << " antennas, K=" << cfg.doa_num_sources
              << " sources, grid=" << cfg.doa_grid_points << " points\n\n";

    int N = cfg.doa_grid_points;
    Eigen::VectorXd theta_grid_deg = Eigen::VectorXd::LinSpaced(N, -90.0, 90.0);
    Eigen::VectorXd theta_grid_rad = theta_grid_deg * M_PI / 180.0;

    std::vector<double> true_doas = {-30.0, 10.0, 45.0};
    std::vector<double> snr_values = {-5, 0, 5, 10, 15, 20, 25, 30};

    // CSV output
    unfolding::CsvWriter csv("data/results/doa_rmse_vs_snr.csv");
    csv.write_header({"snr_db", "ISTA", "FISTA", "MUSIC"});

    for (double snr : snr_values) {
        double ista_rmse_sum = 0.0, fista_rmse_sum = 0.0, music_rmse_sum = 0.0;
        int ista_valid = 0, fista_valid = 0, music_valid = 0;

        for (int trial = 0; trial < cfg.doa_monte_carlo; ++trial) {
            auto prob = unfolding::generate_doa_problem(
                cfg.doa_M, cfg.doa_d_lambda, theta_grid_rad,
                true_doas, snr, trial);

            Eigen::VectorXd y_real = unfolding::complex_to_real_stacked(prob.y);
            Eigen::MatrixXd A_real = unfolding::complex_to_real_stacked(prob.D);

            // ISTA
            {
                unfolding::IstaSolver ista(0.1, 500, 1e-6);
                Eigen::VectorXd x_hat = ista.solve(y_real, A_real);
                Eigen::VectorXd x_mag = extract_magnitudes(x_hat, N);
                auto est = unfolding::extract_doas(x_mag, theta_grid_deg, cfg.doa_num_sources);
                double r = unfolding::doa_rmse(est, true_doas);
                if (std::isfinite(r)) { ista_rmse_sum += r; ++ista_valid; }
            }

            // FISTA
            {
                unfolding::FistaSolver fista(0.1, 200, 1e-6);
                Eigen::VectorXd x_hat = fista.solve(y_real, A_real);
                Eigen::VectorXd x_mag = extract_magnitudes(x_hat, N);
                auto est = unfolding::extract_doas(x_mag, theta_grid_deg, cfg.doa_num_sources);
                double r = unfolding::doa_rmse(est, true_doas);
                if (std::isfinite(r)) { fista_rmse_sum += r; ++fista_valid; }
            }

            // MUSIC
            {
                unfolding::MusicDoa music(cfg.doa_M, cfg.doa_num_sources, cfg.doa_d_lambda);
                // Build sample covariance from single snapshot: R = y * y^H
                Eigen::MatrixXcd R = prob.y * prob.y.adjoint();
                // Add small diagonal for numerical stability
                R += 1e-6 * Eigen::MatrixXcd::Identity(cfg.doa_M, cfg.doa_M);
                auto est = music.estimate_doas(R, theta_grid_rad);
                double r = unfolding::doa_rmse(est, true_doas);
                if (std::isfinite(r)) { music_rmse_sum += r; ++music_valid; }
            }
        }

        double ista_avg = (ista_valid > 0) ? ista_rmse_sum / ista_valid : -1.0;
        double fista_avg = (fista_valid > 0) ? fista_rmse_sum / fista_valid : -1.0;
        double music_avg = (music_valid > 0) ? music_rmse_sum / music_valid : -1.0;

        std::cout << "SNR=" << snr << " dB:  ISTA=" << ista_avg
                  << "  FISTA=" << fista_avg
                  << "  MUSIC=" << music_avg << " deg\n";

        csv.write_row({std::to_string(snr), std::to_string(ista_avg),
                        std::to_string(fista_avg), std::to_string(music_avg)});
    }

    std::cout << "\nResults written to data/results/doa_rmse_vs_snr.csv\n";
    return 0;
}
