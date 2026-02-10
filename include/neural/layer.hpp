#pragma once

#include <Eigen/Dense>
#include "math/soft_threshold.hpp"

namespace unfolding {

/// A single LISTA layer: c_k = soft_threshold(S * c_{k-1} + W_e * y, theta)
struct ListaLayer {
    Eigen::MatrixXd S;       // n × n lateral inhibition
    Eigen::MatrixXd W_e;     // n × m encoder weight
    Eigen::VectorXd theta;   // n thresholds

    Eigen::VectorXd forward(const Eigen::VectorXd& c_prev,
                             const Eigen::VectorXd& y) const {
        return soft_threshold(S * c_prev + W_e * y, theta);
    }
};

/// A single ALISTA layer: c_k = soft_threshold(c_{k-1} - gamma * W * (A*c_{k-1} - y), theta)
struct AlistaLayer {
    double gamma;   // learned step size
    double theta;   // learned threshold (scalar)

    Eigen::VectorXd forward(const Eigen::VectorXd& c_prev,
                             const Eigen::VectorXd& y,
                             const Eigen::MatrixXd& W,
                             const Eigen::MatrixXd& A) const {
        Eigen::VectorXd residual = A * c_prev - y;
        return soft_threshold(c_prev - gamma * (W * residual), theta);
    }
};

}  // namespace unfolding
