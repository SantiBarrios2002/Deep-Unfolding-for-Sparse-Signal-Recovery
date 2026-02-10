#pragma once

#include <Eigen/Dense>
#include <string>

namespace unfolding {

/// Abstract base class for all sparse recovery solvers.
/// Both classical (ISTA, FISTA) and learned (LISTA, ALISTA) inherit this.
class SolverBase {
public:
    virtual ~SolverBase() = default;

    /// Solve: given measurements y and sensing matrix A, recover sparse x̂
    virtual Eigen::VectorXd solve(const Eigen::VectorXd& y,
                                   const Eigen::MatrixXd& A) = 0;

    /// Human-readable solver name (for benchmarking output)
    virtual std::string name() const = 0;
};

}  // namespace unfolding
