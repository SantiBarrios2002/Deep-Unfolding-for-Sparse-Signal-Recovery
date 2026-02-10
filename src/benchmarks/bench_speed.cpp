/// Experiment 3: Speed-Accuracy Benchmark — wall-clock timing comparison

#include <iostream>
#include <vector>

#include "applications/compressed_sensing.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "neural/lista_network.hpp"
#include "evaluation/metrics.hpp"
#include "evaluation/timer.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 3: Speed-Accuracy Benchmark ===\n\n";

    auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, cfg.cs_k,
                                                cfg.cs_snr_db);

    const int num_runs = 100;  // Average over multiple runs for stable timing

    // ISTA timing
    {
        unfolding::IstaSolver ista(cfg.lambda, cfg.ista_max_iter, cfg.solver_tol);
        unfolding::Timer timer("ISTA");
        Eigen::VectorXd x_hat;
        for (int i = 0; i < num_runs; ++i) {
            x_hat = ista.solve(prob.y, prob.A);
        }
        double us = timer.stop() / num_runs;
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "ISTA:  " << us << " us/solve, NMSE = " << nmse << " dB\n";
    }

    // FISTA timing
    {
        unfolding::FistaSolver fista(cfg.lambda, cfg.fista_max_iter, cfg.solver_tol);
        unfolding::Timer timer("FISTA");
        Eigen::VectorXd x_hat;
        for (int i = 0; i < num_runs; ++i) {
            x_hat = fista.solve(prob.y, prob.A);
        }
        double us = timer.stop() / num_runs;
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "FISTA: " << us << " us/solve, NMSE = " << nmse << " dB\n";
    }

    // LISTA (ISTA-init) timing
    {
        auto lista = unfolding::ListaNetwork::from_ista_params(
            prob.A, cfg.lambda, cfg.lista_layers);
        unfolding::Timer timer("LISTA");
        Eigen::VectorXd x_hat;
        for (int i = 0; i < num_runs; ++i) {
            x_hat = lista.solve(prob.y, prob.A);
        }
        double us = timer.stop() / num_runs;
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "LISTA: " << us << " us/solve, NMSE = " << nmse << " dB\n";
    }

    // TODO: Add trained LISTA and ALISTA with loaded weights
    // TODO: Write results to CSV

    std::cout << "\nDone.\n";
    return 0;
}
