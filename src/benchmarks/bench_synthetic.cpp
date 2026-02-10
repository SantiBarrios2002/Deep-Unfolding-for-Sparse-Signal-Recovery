/// Experiment 1: Synthetic Compressed Sensing — NMSE vs iterations/layers
/// Reproduces Gregor & LeCun Fig. 2

#include <iostream>
#include <vector>

#include "applications/compressed_sensing.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "neural/lista_network.hpp"
#include "neural/alista_network.hpp"
#include "neural/weights_loader.hpp"
#include "evaluation/metrics.hpp"
#include "evaluation/timer.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 1: Synthetic Compressed Sensing ===\n";
    std::cout << "n=" << cfg.cs_n << ", m=" << cfg.cs_m
              << ", k=" << cfg.cs_k << ", SNR=" << cfg.cs_snr_db << " dB\n\n";

    // Generate problem
    auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, cfg.cs_k,
                                                cfg.cs_snr_db);

    // ISTA
    {
        unfolding::IstaSolver ista(cfg.lambda, cfg.ista_max_iter, cfg.solver_tol);
        auto x_hat = ista.solve(prob.y, prob.A);
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "ISTA (" << cfg.ista_max_iter << " iter): NMSE = "
                  << nmse << " dB\n";
    }

    // FISTA
    {
        unfolding::FistaSolver fista(cfg.lambda, cfg.fista_max_iter, cfg.solver_tol);
        auto x_hat = fista.solve(prob.y, prob.A);
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "FISTA (" << cfg.fista_max_iter << " iter): NMSE = "
                  << nmse << " dB\n";
    }

    // LISTA (initialized from ISTA params as baseline; replace with trained weights)
    {
        auto lista = unfolding::ListaNetwork::from_ista_params(
            prob.A, cfg.lambda, cfg.lista_layers);
        auto x_hat = lista.solve(prob.y, prob.A);
        double nmse = unfolding::nmse_db(x_hat, prob.x);
        std::cout << "LISTA (" << cfg.lista_layers << " layers, ISTA init): NMSE = "
                  << nmse << " dB\n";
    }

    // TODO: Load trained LISTA weights and re-run
    // TODO: Load trained ALISTA weights and run
    // TODO: Write results to CSV for plotting

    std::cout << "\nDone.\n";
    return 0;
}
