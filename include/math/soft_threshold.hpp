#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace unfolding {

/// Scalar soft-thresholding: sign(u) * max(|u| - alpha, 0)
inline double soft_threshold(double u, double alpha) {
    if (u > alpha) return u - alpha;
    if (u < -alpha) return u + alpha;
    return 0.0;
}

/// Element-wise soft-thresholding with uniform threshold
inline Eigen::VectorXd soft_threshold(const Eigen::VectorXd& u, double alpha) {
    return u.array().sign() * (u.array().abs() - alpha).max(0.0);
}

/// Element-wise soft-thresholding with per-entry thresholds
inline Eigen::VectorXd soft_threshold(const Eigen::VectorXd& u,
                                       const Eigen::VectorXd& alpha) {
    return u.array().sign() * (u.array().abs() - alpha.array()).max(0.0);
}

}  // namespace unfolding
