#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace unfolding {

/// Normalized Mean Squared Error in dB: 10*log10(||x_hat - x||² / ||x||²)
inline double nmse_db(const Eigen::VectorXd& x_hat, const Eigen::VectorXd& x_true) {
    double err = (x_hat - x_true).squaredNorm();
    double sig = x_true.squaredNorm();
    if (sig < 1e-15) return 0.0;
    return 10.0 * std::log10(err / sig);
}

/// Root Mean Squared Error
inline double rmse(const Eigen::VectorXd& x_hat, const Eigen::VectorXd& x_true) {
    return std::sqrt((x_hat - x_true).squaredNorm() / x_hat.size());
}

/// DOA RMSE: RMSE between sorted estimated and true DOA angles (degrees)
inline double doa_rmse(std::vector<double> est, std::vector<double> truth) {
    std::sort(est.begin(), est.end());
    std::sort(truth.begin(), truth.end());
    int K = std::min(est.size(), truth.size());
    double sum = 0.0;
    for (int i = 0; i < K; ++i) {
        double d = est[i] - truth[i];
        sum += d * d;
    }
    return std::sqrt(sum / K);
}

/// Detection probability: fraction of true DOAs detected within ±tol degrees
inline double detection_probability(const std::vector<double>& est,
                                     const std::vector<double>& truth,
                                     double tol = 1.0) {
    int detected = 0;
    for (double t : truth) {
        for (double e : est) {
            if (std::abs(e - t) <= tol) {
                ++detected;
                break;
            }
        }
    }
    return static_cast<double>(detected) / truth.size();
}

}  // namespace unfolding
