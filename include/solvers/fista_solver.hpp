#pragma once

#include "solvers/solver_base.hpp"
#include <vector>

namespace unfolding {

class FistaSolver : public SolverBase {
public:
    FistaSolver(double lambda, int max_iter = 200, double tol = 1e-6);

    Eigen::VectorXd solve(const Eigen::VectorXd& y,
                           const Eigen::MatrixXd& A) override;

    std::string name() const override { return "FISTA"; }

    const std::vector<double>& convergence_history() const { return history_; }

private:
    double lambda_;
    int max_iter_;
    double tol_;
    std::vector<double> history_;
};

}  // namespace unfolding
