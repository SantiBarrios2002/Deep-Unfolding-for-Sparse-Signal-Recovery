/// Experiment 3: Speed-Accuracy Benchmark — wall-clock timing comparison

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
    std::cout << "=== Experiment 3: Speed-Accuracy Benchmark ===\n\n";

    const std::string weights_dir = "data/weights/";
    const std::string A_path = weights_dir + "A.bin";

    auto prob = unfolding::generate_cs_problem(cfg.cs_m, cfg.cs_n, cfg.cs_k,
                                                cfg.cs_snr_db);

    bool have_trained = std::filesystem::exists(A_path);
    if (have_trained) {
        Eigen::MatrixXd A = unfolding::load_matrix_binary(A_path);
        prob.A = A;
        prob.y = A * prob.x;
        std::mt19937 gen(44);
        std::normal_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd noise(cfg.cs_m);
        for (int i = 0; i < cfg.cs_m; ++i) noise(i) = dist(gen);
        double signal_power = prob.y.squaredNorm() / cfg.cs_m;
        double noise_power = signal_power * std::pow(10.0, -cfg.cs_snr_db / 10.0);
        noise *= std::sqrt(noise_power / (noise.squaredNorm() / cfg.cs_m));
        prob.y += noise;
    }

    const int num_runs = 100;

    unfolding::CsvWriter csv("data/results/bench_speed.csv");
    csv.write_header({"method", "iterations_or_layers", "nmse_db", "time_us"});

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
        std::cout << "ISTA:   " << us << " us/solve, NMSE = " << nmse << " dB\n";
        csv.write_row({"ISTA", std::to_string(cfg.ista_max_iter),
                        std::to_string(nmse), std::to_string(us)});
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
        std::cout << "FISTA:  " << us << " us/solve, NMSE = " << nmse << " dB\n";
        csv.write_row({"FISTA", std::to_string(cfg.fista_max_iter),
                        std::to_string(nmse), std::to_string(us)});
    }

    // LISTA (trained) timing
    if (have_trained && std::filesystem::exists(weights_dir + "lista.bin")) {
        try {
            auto lista = unfolding::load_lista_weights(weights_dir + "lista.bin");
            unfolding::Timer timer("LISTA");
            Eigen::VectorXd x_hat;
            for (int i = 0; i < num_runs; ++i) {
                x_hat = lista.solve(prob.y, prob.A);
            }
            double us = timer.stop() / num_runs;
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            std::cout << "LISTA:  " << us << " us/solve, NMSE = " << nmse << " dB\n";
            csv.write_row({"LISTA", std::to_string(lista.num_layers()),
                            std::to_string(nmse), std::to_string(us)});
        } catch (const std::exception& e) {
            std::cerr << "LISTA: " << e.what() << "\n";
        }
    }

    // ALISTA (trained) timing
    if (have_trained && std::filesystem::exists(weights_dir + "alista.bin")) {
        try {
            auto params = unfolding::load_alista_weights(weights_dir + "alista.bin");
            unfolding::AlistaNetwork alista(prob.A, params.gamma, params.theta);
            unfolding::Timer timer("ALISTA");
            Eigen::VectorXd x_hat;
            for (int i = 0; i < num_runs; ++i) {
                x_hat = alista.solve(prob.y, prob.A);
            }
            double us = timer.stop() / num_runs;
            double nmse = unfolding::nmse_db(x_hat, prob.x);
            std::cout << "ALISTA: " << us << " us/solve, NMSE = " << nmse << " dB\n";
            csv.write_row({"ALISTA", std::to_string(alista.num_layers()),
                            std::to_string(nmse), std::to_string(us)});
        } catch (const std::exception& e) {
            std::cerr << "ALISTA: " << e.what() << "\n";
        }
    }

    std::cout << "\nResults written to data/results/bench_speed.csv\n";
    return 0;
}
