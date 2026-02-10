/// Experiment 4: Audio Inpainting Demo

#include <iostream>

#include "applications/audio_inpainter.hpp"
#include "solvers/ista_solver.hpp"
#include "solvers/fista_solver.hpp"
#include "evaluation/metrics.hpp"
#include "evaluation/timer.hpp"
#include "utils/config.hpp"

int main() {
    unfolding::Config cfg;
    std::cout << "=== Experiment 4: Audio Inpainting ===\n\n";

    // TODO: Load audio from data/audio/ using wav_io
    // For now, generate a synthetic test signal (sum of sinusoids)
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

    // Recover with ISTA
    {
        unfolding::IstaSolver ista(0.01, 200, 1e-5);
        unfolding::Timer timer("ISTA recovery");
        auto recovered = inpainter.recover(corrupted, mask, ista);
        double us = timer.stop();

        double nmse = unfolding::nmse_db(recovered, audio);
        std::cout << "ISTA recovery: NMSE = " << nmse << " dB, time = "
                  << us / 1e6 << " s\n";

        // TODO: Write recovered audio to data/audio/recovered_ista.wav
    }

    // TODO: Recover with LISTA (trained weights)
    // TODO: Write all .wav files for presentation demo

    std::cout << "\nDone.\n";
    return 0;
}
