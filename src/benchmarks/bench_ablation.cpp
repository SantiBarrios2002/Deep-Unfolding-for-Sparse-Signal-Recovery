/// Experiment 5: Generalization & Ablation Studies

#include <iostream>
#include <vector>
#include <filesystem>
#include <random>
#include <cmath>

#include "applications/compressed_sensing.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "neural/lista_network.hpp"
#include "neural/alista_network.hpp"
#include "neural/weights_loader.hpp"
#include "evaluation/metrics.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 5: Generalization & Ablation ===\n\n";

    const std::string weights_dir = "data/weights/";
    bool have_trained = std::filesystem::exists(weights_dir + "A.bin");

    Eigen::MatrixXd A_trained;
    if (have_trained) {
        A_trained = unfolding::load_matrix_binary(weights_dir + "A.bin");
    }

    // ---- (a) Sparsity generalization ----
    std::cout << "--- 5a: Sparsity Generalization ---\n";
    {
        unfolding::CsvWriter csv("data/results/ablation_sparsity.csv");
        csv.write_header({"sparsity", "ISTA", "FISTA", "LISTA", "ALISTA"});

        std::vector<int> sparsity_levels = {10, 15, 20, 25, 30, 40, 50};
        for (int k : sparsity_levels) {
            auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, k,
                                                        cfg.cs_snr_db);
            if (have_trained) {
                prob.A = A_trained;
                prob.y = A_trained * prob.x;
            }

            // ISTA
            unfolding::IstaSolver ista(cfg.lambda, cfg.ista_max_iter, cfg.solver_tol);
            double nmse_ista = unfolding::nmse_db(ista.solve(prob.y, prob.A), prob.x);

            // FISTA
            unfolding::FistaSolver fista(cfg.lambda, cfg.fista_max_iter, cfg.solver_tol);
            double nmse_fista = unfolding::nmse_db(fista.solve(prob.y, prob.A), prob.x);

            // LISTA
            double nmse_lista = 0.0;
            if (have_trained && std::filesystem::exists(weights_dir + "lista.bin")) {
                try {
                    auto lista = unfolding::load_lista_weights(weights_dir + "lista.bin");
                    nmse_lista = unfolding::nmse_db(lista.solve(prob.y, prob.A), prob.x);
                } catch (...) { nmse_lista = 0.0; }
            }

            // ALISTA
            double nmse_alista = 0.0;
            if (have_trained && std::filesystem::exists(weights_dir + "alista.bin")) {
                try {
                    auto params = unfolding::load_alista_weights(weights_dir + "alista.bin");
                    unfolding::AlistaNetwork alista(prob.A, params.gamma, params.theta);
                    nmse_alista = unfolding::nmse_db(alista.solve(prob.y, prob.A), prob.x);
                } catch (...) { nmse_alista = 0.0; }
            }

            std::cout << "  k=" << k << ": ISTA=" << nmse_ista
                      << " FISTA=" << nmse_fista
                      << " LISTA=" << nmse_lista
                      << " ALISTA=" << nmse_alista << " dB\n";

            csv.write_row({std::to_string(k), std::to_string(nmse_ista),
                            std::to_string(nmse_fista), std::to_string(nmse_lista),
                            std::to_string(nmse_alista)});
        }
    }

    // ---- (b) Layer count sweep ----
    std::cout << "\n--- 5b: Layer Count Sweep ---\n";
    {
        unfolding::CsvWriter csv("data/results/ablation_layers.csv");
        csv.write_header({"layers", "LISTA_ista_init"});

        auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, cfg.cs_k,
                                                    cfg.cs_snr_db);
        if (have_trained) {
            prob.A = A_trained;
            prob.y = A_trained * prob.x;
        }

        std::vector<int> layer_counts = {2, 4, 8, 12, 16, 20, 24};
        for (int K : layer_counts) {
            auto lista = unfolding::ListaNetwork::from_ista_params(
                prob.A, cfg.lambda, K);
            auto x_hat = lista.solve(prob.y, prob.A);
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            std::cout << "  K=" << K << " layers: NMSE = " << nmse << " dB\n";
            csv.write_row({std::to_string(K), std::to_string(nmse)});
        }
    }

    // ---- (c) Parameter count comparison ----
    std::cout << "\n--- 5c: Parameter Count ---\n";
    {
        int n = cfg.cs_n, m = cfg.cs_m, K = cfg.lista_layers;
        long lista_params = static_cast<long>(K) * (n * n + n * m + n);
        long alista_params = static_cast<long>(K) * 2;
        std::cout << "  LISTA:  " << lista_params << " parameters\n";
        std::cout << "  ALISTA: " << alista_params << " parameters\n";
        std::cout << "  Ratio:  " << lista_params / alista_params << "x\n";
    }

    // ---- (d) SNR robustness ----
    std::cout << "\n--- 5d: SNR Robustness ---\n";
    {
        unfolding::CsvWriter csv("data/results/ablation_snr.csv");
        csv.write_header({"snr_db", "ISTA", "FISTA", "LISTA", "ALISTA"});

        std::vector<double> snr_values = {10, 20, 30, 40};
        for (double snr : snr_values) {
            auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n,
                                                        cfg.cs_k, snr);
            if (have_trained) {
                prob.A = A_trained;
                prob.y = A_trained * prob.x;
                // Re-add noise at this SNR
                std::mt19937 gen(44);
                std::normal_distribution<double> dist(0.0, 1.0);
                Eigen::VectorXd noise(cfg.cs_m);
                for (int i = 0; i < cfg.cs_m; ++i) noise(i) = dist(gen);
                double sp = prob.y.squaredNorm() / cfg.cs_m;
                double np = sp * std::pow(10.0, -snr / 10.0);
                noise *= std::sqrt(np / (noise.squaredNorm() / cfg.cs_m));
                prob.y += noise;
            }

            unfolding::IstaSolver ista(cfg.lambda, cfg.ista_max_iter, cfg.solver_tol);
            double nmse_ista = unfolding::nmse_db(ista.solve(prob.y, prob.A), prob.x);

            unfolding::FistaSolver fista(cfg.lambda, cfg.fista_max_iter, cfg.solver_tol);
            double nmse_fista = unfolding::nmse_db(fista.solve(prob.y, prob.A), prob.x);

            double nmse_lista = 0.0, nmse_alista = 0.0;
            if (have_trained) {
                try {
                    auto lista = unfolding::load_lista_weights(weights_dir + "lista.bin");
                    nmse_lista = unfolding::nmse_db(lista.solve(prob.y, prob.A), prob.x);
                } catch (...) {}
                try {
                    auto params = unfolding::load_alista_weights(weights_dir + "alista.bin");
                    unfolding::AlistaNetwork alista(prob.A, params.gamma, params.theta);
                    nmse_alista = unfolding::nmse_db(alista.solve(prob.y, prob.A), prob.x);
                } catch (...) {}
            }

            std::cout << "  SNR=" << snr << " dB: ISTA=" << nmse_ista
                      << " FISTA=" << nmse_fista
                      << " LISTA=" << nmse_lista
                      << " ALISTA=" << nmse_alista << " dB\n";

            csv.write_row({std::to_string(snr), std::to_string(nmse_ista),
                            std::to_string(nmse_fista), std::to_string(nmse_lista),
                            std::to_string(nmse_alista)});
        }
    }

    std::cout << "\nDone. Results written to data/results/ablation_*.csv\n";
    return 0;
}
