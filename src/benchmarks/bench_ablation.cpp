/// Experiment 5: Generalization & Ablation Studies

#include <iostream>
#include <vector>

#include "applications/compressed_sensing.hpp"
#include "solvers/ista_solver.hpp"
#include "neural/lista_network.hpp"
#include "evaluation/metrics.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 5: Generalization & Ablation ===\n\n";

    // (a) Sparsity generalization: train on k=25, test on various k
    std::cout << "--- 5a: Sparsity Generalization ---\n";
    std::vector<int> sparsity_levels = {10, 15, 20, 25, 30, 40, 50};
    for (int k : sparsity_levels) {
        auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, k,
                                                    cfg.cs_snr_db);
        unfolding::IstaSolver ista(cfg.lambda, cfg.ista_max_iter, cfg.solver_tol);
        auto x_hat = ista.solve(prob.y, prob.A);
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "  k=" << k << ": ISTA NMSE = " << nmse << " dB\n";
    }

    // (b) Layer count sweep
    std::cout << "\n--- 5b: Layer Count Sweep ---\n";
    auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, cfg.cs_k,
                                                cfg.cs_snr_db);
    std::vector<int> layer_counts = {2, 4, 8, 12, 16, 20, 24};
    for (int K : layer_counts) {
        auto lista = unfolding::ListaNetwork::from_ista_params(
            prob.A, cfg.lambda, K);
        auto x_hat = lista.solve(prob.y, prob.A);
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "  K=" << K << " layers: LISTA (ISTA init) NMSE = "
                  << nmse << " dB\n";
    }

    // (c) Parameter count comparison
    std::cout << "\n--- 5c: Parameter Count ---\n";
    int n = cfg.cs_n, m = cfg.cs_m, K = cfg.lista_layers;
    long lista_params = static_cast<long>(K) * (n * n + n * m + n);
    long alista_params = static_cast<long>(K) * 2;
    std::cout << "  LISTA:  " << lista_params << " parameters\n";
    std::cout << "  ALISTA: " << alista_params << " parameters\n";
    std::cout << "  Ratio:  " << lista_params / alista_params << "x\n";

    // (d) SNR robustness
    std::cout << "\n--- 5d: SNR Robustness ---\n";
    std::vector<double> snr_values = {10, 20, 30, 40};
    for (double snr : snr_values) {
        auto prob_snr = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n,
                                                        cfg.cs_k, snr);
        unfolding::IstaSolver ista(cfg.lambda, cfg.ista_max_iter, cfg.solver_tol);
        auto x_hat = ista.solve(prob_snr.y, prob_snr.A);
        double nmse = unfolding::nmse_db(x_hat, prob_snr.x);
        std::cout << "  SNR=" << snr << " dB: ISTA NMSE = " << nmse << " dB\n";
    }

    // TODO: Add trained LISTA/ALISTA comparisons
    // TODO: Write all results to CSV for plotting

    std::cout << "\nDone.\n";
    return 0;
}
