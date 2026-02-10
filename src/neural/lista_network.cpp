#include "neural/lista_network.hpp"
#include "math/soft_threshold.hpp"
#include "math/matrix_utils.hpp"

namespace unfolding {

ListaNetwork::ListaNetwork(std::vector<ListaLayer> layers)
    : layers_(std::move(layers)) {}

Eigen::VectorXd ListaNetwork::solve(const Eigen::VectorXd& y,
                                      const Eigen::MatrixXd& /*A*/) {
    // Layer 0: c = soft_threshold(W_e[0] * y, theta[0])
    Eigen::VectorXd c = soft_threshold(layers_[0].W_e * y, layers_[0].theta);

    // Layers 1 to K-1: c = soft_threshold(S[k]*c + W_e[k]*y, theta[k])
    for (size_t k = 1; k < layers_.size(); ++k) {
        c = layers_[k].forward(c, y);
    }

    return c;
}

ListaNetwork ListaNetwork::from_ista_params(const Eigen::MatrixXd& A,
                                              double lambda, int K) {
    int n = A.cols();
    double L = compute_lipschitz_constant(A);

    // ISTA-equivalent parameters: S = I - (1/L)*A^T*A, W_e = (1/L)*A^T
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n)
                        - (1.0 / L) * (A.transpose() * A);
    Eigen::MatrixXd W_e = (1.0 / L) * A.transpose();
    Eigen::VectorXd theta = Eigen::VectorXd::Constant(n, lambda / L);

    std::vector<ListaLayer> layers(K);
    for (int k = 0; k < K; ++k) {
        layers[k].S = S;
        layers[k].W_e = W_e;
        layers[k].theta = theta;
    }

    return ListaNetwork(std::move(layers));
}

}  // namespace unfolding
