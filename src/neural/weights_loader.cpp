#include "neural/weights_loader.hpp"
#include <fstream>
#include <stdexcept>

namespace unfolding {

ListaNetwork load_lista_weights(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + filepath);
    }

    int K, n, m;
    f.read(reinterpret_cast<char*>(&K), sizeof(int));
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    f.read(reinterpret_cast<char*>(&m), sizeof(int));

    std::vector<ListaLayer> layers(K);
    for (int k = 0; k < K; ++k) {
        layers[k].S.resize(n, n);
        f.read(reinterpret_cast<char*>(layers[k].S.data()), n * n * sizeof(double));

        layers[k].W_e.resize(n, m);
        f.read(reinterpret_cast<char*>(layers[k].W_e.data()), n * m * sizeof(double));

        layers[k].theta.resize(n);
        f.read(reinterpret_cast<char*>(layers[k].theta.data()), n * sizeof(double));
    }

    return ListaNetwork(std::move(layers));
}

AlistaParams load_alista_weights(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + filepath);
    }

    int K;
    f.read(reinterpret_cast<char*>(&K), sizeof(int));

    AlistaParams params;
    params.gamma.resize(K);
    params.theta.resize(K);

    for (int k = 0; k < K; ++k) {
        f.read(reinterpret_cast<char*>(&params.gamma[k]), sizeof(double));
        f.read(reinterpret_cast<char*>(&params.theta[k]), sizeof(double));
    }

    return params;
}

Eigen::MatrixXd load_matrix_binary(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open matrix file: " + filepath);
    }

    int m, n;
    f.read(reinterpret_cast<char*>(&m), sizeof(int));
    f.read(reinterpret_cast<char*>(&n), sizeof(int));

    Eigen::MatrixXd A(m, n);
    f.read(reinterpret_cast<char*>(A.data()), m * n * sizeof(double));

    return A;
}

}  // namespace unfolding
