/// Experiment 4: Audio Inpainting Demo

#include <iostream>
#include <cmath>

#include "applications/audio_inpainter.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "evaluation/metrics.hpp"
#include "evaluation/timer.hpp"
#include "utils/csv_writer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 4: Audio Inpainting ===\n\n";

    // Generate a synthetic test signal (sum of sinusoids — sparse in DCT domain)
    int N = cfg.audio_sample_rate * 3;  // 3 seconds
    Eigen::VectorXd audio(N);
    for (int i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / cfg.audio_sample_rate;
        audio(i) = 0.5 * std::sin(2.0 * M_PI * 440.0 * t)   // A4
                  + 0.3 * std::sin(2.0 * M_PI * 880.0 * t)   // A5
                  + 0.2 * std::sin(2.0 * M_PI * 1320.0 * t); // E6
    }

    unfolding::AudioInpainter inpainter(cfg.audio_frame_size, cfg.audio_overlap);

    // Corrupt
    Eigen::VectorXi mask;
    auto corrupted = inpainter.corrupt(audio, cfg.audio_drop_fraction, mask);
    std::cout << "Original samples: " << N << "\n";
    std::cout << "Retained samples: " << mask.sum() << " ("
              << 100.0 * mask.sum() / N << "%)\n\n";

    // CSV output
    unfolding::CsvWriter csv("data/results/bench_audio.csv");
    csv.write_header({"method", "nmse_db", "time_s"});

    // Recover with ISTA
    {
        unfolding::IstaSolver ista(0.01, 200, 1e-5);
        unfolding::Timer timer("ISTA recovery");
        auto recovered = inpainter.recover(corrupted, mask, ista);
        double time_s = timer.stop() / 1e6;

        double nmse = unfolding::nmse_db(recovered, audio);
        std::cout << "ISTA recovery:  NMSE = " << nmse << " dB, time = "
                  << time_s << " s\n";
        csv.write_row({"ISTA", std::to_string(nmse), std::to_string(time_s)});
    }

    // Recover with FISTA
    {
        unfolding::FistaSolver fista(0.01, 100, 1e-5);
        unfolding::Timer timer("FISTA recovery");
        auto recovered = inpainter.recover(corrupted, mask, fista);
        double time_s = timer.stop() / 1e6;

        double nmse = unfolding::nmse_db(recovered, audio);
        std::cout << "FISTA recovery: NMSE = " << nmse << " dB, time = "
                  << time_s << " s\n";
        csv.write_row({"FISTA", std::to_string(nmse), std::to_string(time_s)});
    }

    std::cout << "\nResults written to data/results/bench_audio.csv\n";
    return 0;
}
