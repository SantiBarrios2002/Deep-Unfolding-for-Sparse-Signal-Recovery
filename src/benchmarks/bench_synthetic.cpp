/// Experiment 1: Synthetic Compressed Sensing — NMSE vs iterations/layers
/// Reproduces Gregor & LeCun Fig. 2

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
#include "evaluation/timer.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 1: Synthetic Compressed Sensing ===\n";
    std::cout << "n=" << cfg.cs_n << ", m=" << cfg.cs_m
              << ", k=" << cfg.cs_k << ", SNR=" << cfg.cs_snr_db << " dB\n\n";

    // Load the sensing matrix used during training
    const std::string weights_dir = "data/weights/";
    const std::string A_path = weights_dir + "A.bin";

    Eigen::MatrixXd A;
    bool have_trained = std::filesystem::exists(A_path);
    if (have_trained) {
        A = unfolding::load_matrix_binary(A_path);
        std::cout << "Loaded training A: " << A.rows() << "x" << A.cols() << "\n\n";
    } else {
        std::cout << "Warning: " << A_path << " not found. Using random A.\n\n";
    }

    // Generate test signal using the loaded (or generated) A
    auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, cfg.cs_k,
                                                cfg.cs_snr_db);
    if (have_trained) {
        prob.A = A;
        prob.y = A * prob.x;
        // Re-add noise
        std::mt19937 gen(44);
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd noise(cfg.cs_m);
        for (int i = 0; i < cfg.cs_m; ++i) noise(i) = dist(gen);
        double signal_power = prob.y.squaredNorm() / cfg.cs_m;
        double noise_power = signal_power * std::pow(10.0, -cfg.cs_snr_db / 10.0);
        noise *= std::sqrt(noise_power / (noise.squaredNorm() / cfg.cs_m));
        prob.y += noise;
    }

    // CSV output
    unfolding::CsvWriter csv("data/results/bench_synthetic.csv");
    csv.write_header({"method", "iterations_or_layers", "nmse_db"});

    // ISTA — sweep iterations
    {
        std::vector<int> iter_counts = {1, 2, 5, 10, 20, 50, 100, 200, 500};
        for (int iters : iter_counts) {
            unfolding::IstaSolver ista(cfg.lambda, iters, 0.0);
            auto x_hat = ista.solve(prob.y, prob.A);
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            csv.write_row({"ISTA", std::to_string(iters), std::to_string(nmse)});
            if (iters == 500)
                std::cout << "ISTA (500 iter): NMSE = " << nmse << " dB\n";
        }
    }

    // FISTA — sweep iterations
    {
        std::vector<int> iter_counts = {1, 2, 5, 10, 20, 50, 100, 200};
        for (int iters : iter_counts) {
            unfolding::FistaSolver fista(cfg.lambda, iters, 0.0);
            auto x_hat = fista.solve(prob.y, prob.A);
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            csv.write_row({"FISTA", std::to_string(iters), std::to_string(nmse)});
            if (iters == 200)
                std::cout << "FISTA (200 iter): NMSE = " << nmse << " dB\n";
        }
    }

    // LISTA with trained weights
    if (have_trained && std::filesystem::exists(weights_dir + "lista.bin")) {
        try {
            auto lista = unfolding::load_lista_weights(weights_dir + "lista.bin");
            // Sweep layers by using partial evaluation (run K layers = full network)
            auto x_hat = lista.solve(prob.y, prob.A);
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            csv.write_row({"LISTA", std::to_string(lista.num_layers()), std::to_string(nmse)});
            std::cout << "LISTA (" << lista.num_layers() << " layers, trained): NMSE = "
                      << nmse << " dB\n";
        } catch (const std::exception& e) {
            std::cerr << "LISTA load failed: " << e.what() << "\n";
        }
    }

    // ALISTA with trained weights
    if (have_trained && std::filesystem::exists(weights_dir + "alista.bin")) {
        try {
            auto params = unfolding::load_alista_weights(weights_dir + "alista.bin");
            unfolding::AlistaNetwork alista(prob.A, params.gamma, params.theta);
            auto x_hat = alista.solve(prob.y, prob.A);
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            csv.write_row({"ALISTA", std::to_string(alista.num_layers()), std::to_string(nmse)});
            std::cout << "ALISTA (" << alista.num_layers() << " layers, trained): NMSE = "
                      << nmse << " dB\n";
        } catch (const std::exception& e) {
            std::cerr << "ALISTA load failed: " << e.what() << "\n";
        }
    }

    std::cout << "\nResults written to data/results/bench_synthetic.csv\n";
    return 0;
}
