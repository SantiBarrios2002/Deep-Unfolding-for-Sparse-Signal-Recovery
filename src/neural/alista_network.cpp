#include "neural/alista_network.hpp"
#include "math/soft_threshold.hpp"

namespace unfolding {

AlistaNetwork::AlistaNetwork(const Eigen::MatrixXd& A,
                               const std::vector<double>& gamma,
                               const std::vector<double>& theta,
                               double alpha)
    : A_(A) {
    W_ = compute_analytic_weights(A, alpha);

    int K = static_cast<int>(gamma.size());
    layers_.resize(K);
    for (int k = 0; k < K; ++k) {
        layers_[k].gamma = gamma[k];
        layers_[k].theta = theta[k];
    }
}

Eigen::VectorXd AlistaNetwork::solve(const Eigen::VectorXd& y,
                                       const Eigen::MatrixXd& /*A*/) {
    int n = A_.cols();
    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);

    for (size_t k = 0; k < layers_.size(); ++k) {
        c = layers_[k].forward(c, y, W_, A_);
    }

    return c;
}

Eigen::MatrixXd AlistaNetwork::compute_analytic_weights(const Eigen::MatrixXd& A,
                                                          double alpha) {
    // W = (A^T A + alpha * I)^{-1} A^T
    int n = A.cols();
    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::MatrixXd reg = AtA + alpha * Eigen::MatrixXd::Identity(n, n);
    return reg.ldlt().solve(A.transpose());
}

}  // namespace unfolding
