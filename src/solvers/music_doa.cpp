#include "solvers/music_doa.hpp"
#include <algorithm>
#include <cmath>
#include <complex>

namespace unfolding {

MusicDoa::MusicDoa(int M, int num_sources, double d_lambda)
    : M_(M), num_sources_(num_sources), d_lambda_(d_lambda) {}

Eigen::VectorXd MusicDoa::compute_spectrum(const Eigen::MatrixXcd& R,
                                             const Eigen::VectorXd& theta_grid) const {
    // Eigendecomposition of covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(R);
    Eigen::MatrixXcd eigvecs = solver.eigenvectors();

    // Noise subspace: eigenvectors corresponding to smallest eigenvalues
    Eigen::MatrixXcd Un = eigvecs.leftCols(M_ - num_sources_);

    int N = theta_grid.size();
    Eigen::VectorXd spectrum(N);

    for (int j = 0; j < N; ++j) {
        // Build steering vector for this angle
        Eigen::VectorXcd a(M_);
        for (int i = 0; i < M_; ++i) {
            double phase = 2.0 * M_PI * i * d_lambda_ * std::sin(theta_grid(j));
            a(i) = std::complex<double>(std::cos(phase), std::sin(phase));
        }

        // MUSIC pseudo-spectrum: 1 / (a^H * Un * Un^H * a)
        Eigen::VectorXcd proj = Un.adjoint() * a;
        double denom = proj.squaredNorm();
        spectrum(j) = (denom > 1e-15) ? 1.0 / denom : 1e15;
    }

    return spectrum;
}

std::vector<double> MusicDoa::estimate_doas(const Eigen::MatrixXcd& R,
                                              const Eigen::VectorXd& theta_grid) const {
    Eigen::VectorXd spectrum = compute_spectrum(R, theta_grid);

    // Find peaks: local maxima in the spectrum
    std::vector<std::pair<double, double>> peaks;  // (spectrum value, angle)
    for (int j = 1; j < spectrum.size() - 1; ++j) {
        if (spectrum(j) > spectrum(j - 1) && spectrum(j) > spectrum(j + 1)) {
            peaks.emplace_back(spectrum(j), theta_grid(j));
        }
    }

    // Sort by spectrum value (descending) and take top num_sources_
    std::sort(peaks.begin(), peaks.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<double> doas;
    int K = std::min(num_sources_, static_cast<int>(peaks.size()));
    for (int i = 0; i < K; ++i) {
        doas.push_back(peaks[i].second * 180.0 / M_PI);  // Convert to degrees
    }

    return doas;
}

}  // namespace unfolding
