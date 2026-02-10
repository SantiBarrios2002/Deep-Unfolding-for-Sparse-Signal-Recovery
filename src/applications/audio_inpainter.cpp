#include "applications/audio_inpainter.hpp"
#include "solvers/solver_base.hpp"
#include <cmath>
#include <random>

namespace unfolding {

AudioInpainter::AudioInpainter(int frame_size, int overlap)
    : frame_size_(frame_size), overlap_(overlap) {}

Eigen::VectorXd AudioInpainter::corrupt(const Eigen::VectorXd& audio,
                                          double drop_fraction,
                                          Eigen::VectorXi& mask,
                                          unsigned int seed) const {
    int N = audio.size();
    mask = Eigen::VectorXi::Ones(N);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        if (dist(gen) < drop_fraction) {
            mask(i) = 0;
        }
    }

    // Apply mask: zero out dropped samples
    return audio.array() * mask.cast<double>().array();
}

Eigen::VectorXd AudioInpainter::recover(const Eigen::VectorXd& corrupted,
                                          const Eigen::VectorXi& mask,
                                          SolverBase& solver) const {
    int N = corrupted.size();
    int hop = frame_size_ - overlap_;
    Eigen::MatrixXd dct = build_dct_matrix(frame_size_);

    Eigen::VectorXd recovered = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd weight = Eigen::VectorXd::Zero(N);

    // Frame-wise processing with overlap-add
    for (int start = 0; start + frame_size_ <= N; start += hop) {
        // Extract frame and its mask
        Eigen::VectorXd frame = corrupted.segment(start, frame_size_);
        Eigen::VectorXi frame_mask = mask.segment(start, frame_size_);

        // Build sampling operator: rows of identity where mask = 1
        int m = frame_mask.sum();
        Eigen::MatrixXd sampling(m, frame_size_);
        int row = 0;
        for (int i = 0; i < frame_size_; ++i) {
            if (frame_mask(i)) {
                sampling.row(row) = Eigen::RowVectorXd::Unit(frame_size_, i);
                ++row;
            }
        }

        // Sensing matrix: sampling operator × DCT
        Eigen::MatrixXd A = sampling * dct;
        Eigen::VectorXd y = sampling * frame;

        // Recover DCT coefficients via sparse solver
        Eigen::VectorXd x_hat = solver.solve(y, A);

        // Reconstruct frame from DCT coefficients
        Eigen::VectorXd frame_recovered = dct * x_hat;

        // Overlap-add
        recovered.segment(start, frame_size_) += frame_recovered;
        weight.segment(start, frame_size_) += Eigen::VectorXd::Ones(frame_size_);
    }

    // Normalize overlap regions
    for (int i = 0; i < N; ++i) {
        if (weight(i) > 0) recovered(i) /= weight(i);
    }

    return recovered;
}

Eigen::MatrixXd AudioInpainter::build_dct_matrix(int N) {
    Eigen::MatrixXd dct(N, N);
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            dct(n, k) = std::cos(M_PI * (n + 0.5) * k / N);
        }
    }
    // Normalize columns
    dct.col(0) *= std::sqrt(1.0 / N);
    for (int k = 1; k < N; ++k) {
        dct.col(k) *= std::sqrt(2.0 / N);
    }
    return dct;
}

}  // namespace unfolding
