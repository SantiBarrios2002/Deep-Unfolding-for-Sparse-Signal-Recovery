#pragma once

#include "solvers/solver_base.hpp"
#include "neural/layer.hpp"
#include <vector>

namespace unfolding {

class ListaNetwork : public SolverBase {
public:
    /// Construct from pre-loaded layers
    explicit ListaNetwork(std::vector<ListaLayer> layers);

    Eigen::VectorXd solve(const Eigen::VectorXd& y,
                           const Eigen::MatrixXd& A) override;

    std::string name() const override { return "LISTA"; }

    int num_layers() const { return static_cast<int>(layers_.size()); }

    /// Factory: initialize LISTA with ISTA parameters (for testing equivalence)
    static ListaNetwork from_ista_params(const Eigen::MatrixXd& A,
                                          double lambda, int K);

private:
    std::vector<ListaLayer> layers_;
};

}  // namespace unfolding
