#pragma once

namespace unfolding {

/// Default experiment parameters from CLAUDE.md
struct Config {
    // Experiment 1: Synthetic Compressed Sensing
    int cs_n = 500;            // signal dimension
    int cs_m = 250;            // measurement dimension
    int cs_k = 25;             // sparsity level
    double cs_snr_db = 40.0;   // signal-to-noise ratio

    // Solver parameters
    int ista_max_iter = 500;
    int fista_max_iter = 200;
    double solver_tol = 1e-6;
    double lambda = 0.1;       // regularization parameter

    // Network parameters
    int lista_layers = 16;
    int alista_layers = 16;
    double alista_alpha = 1e-3;  // regularization for analytic weight computation

    // Experiment 2: DOA Estimation
    int doa_M = 16;              // number of antennas
    double doa_d_lambda = 0.5;   // antenna spacing / wavelength
    int doa_num_sources = 3;
    int doa_grid_points = 181;   // 1° resolution from -90° to 90°
    int doa_monte_carlo = 100;

    // Experiment 4: Audio Inpainting
    int audio_frame_size = 512;
    int audio_overlap = 256;
    int audio_sample_rate = 16000;
    double audio_drop_fraction = 0.5;
};

}  // namespace unfolding
