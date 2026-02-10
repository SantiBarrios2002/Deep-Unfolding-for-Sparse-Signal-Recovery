#pragma once

#include "solvers/solver_base.hpp"
#include "neural/layer.hpp"
#include <vector>

namespace unfolding {

class AlistaNetwork : public SolverBase {
public:
    /// Construct from sensing matrix A, learned step sizes and thresholds
    AlistaNetwork(const Eigen::MatrixXd& A,
                   const std::vector<double>& gamma,
                   const std::vector<double>& theta,
                   double alpha = 1e-3);

    Eigen::VectorXd solve(const Eigen::VectorXd& y,
                           const Eigen::MatrixXd& A) override;

    std::string name() const override { return "ALISTA"; }

    int num_layers() const { return static_cast<int>(layers_.size()); }

    /// Compute W = (A^T A + alpha I)^{-1} A^T analytically
    static Eigen::MatrixXd compute_analytic_weights(const Eigen::MatrixXd& A,
                                                      double alpha = 1e-3);

private:
    Eigen::MatrixXd W_;  // analytic encoder weight (precomputed)
    Eigen::MatrixXd A_;  // sensing matrix (stored for forward pass)
    std::vector<AlistaLayer> layers_;
};

}  // namespace unfolding
