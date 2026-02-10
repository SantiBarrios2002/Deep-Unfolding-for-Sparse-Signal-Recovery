/// Test round-trip: write weights binary → load back → verify match

#include "neural/weights_loader.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <filesystem>

void assert_near(double a, double b, double tol = 1e-12) {
    if (std::abs(a - b) > tol) {
        std::cerr << "FAIL: " << a << " != " << b << "\n";
        assert(false);
    }
}

int main() {
    // Create small test weights
    int K = 3, n = 4, m = 2;

    std::vector<Eigen::MatrixXd> S_orig(K), W_e_orig(K);
    std::vector<Eigen::VectorXd> theta_orig(K);

    for (int k = 0; k < K; ++k) {
        S_orig[k] = Eigen::MatrixXd::Random(n, n);
        W_e_orig[k] = Eigen::MatrixXd::Random(n, m);
        theta_orig[k] = Eigen::VectorXd::Random(n).cwiseAbs();
    }

    // Write to binary
    std::string path = "test_weights_tmp.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&K), sizeof(int));
        f.write(reinterpret_cast<const char*>(&n), sizeof(int));
        f.write(reinterpret_cast<const char*>(&m), sizeof(int));
        for (int k = 0; k < K; ++k) {
            f.write(reinterpret_cast<const char*>(S_orig[k].data()),
                    n * n * sizeof(double));
            f.write(reinterpret_cast<const char*>(W_e_orig[k].data()),
                    n * m * sizeof(double));
            f.write(reinterpret_cast<const char*>(theta_orig[k].data()),
                    n * sizeof(double));
        }
    }

    // Load back using the loader
    auto lista = unfolding::load_lista_weights(path);
    assert(lista.num_layers() == K);

    // Verify by running a forward pass and checking consistency
    // (Direct layer access would need a getter; for now just verify it loaded)
    std::cout << "Loaded " << lista.num_layers() << " layers successfully.\n";

    // Test ALISTA round-trip
    std::string alista_path = "test_alista_weights_tmp.bin";
    std::vector<double> gamma_orig = {0.5, 0.3, 0.1};
    std::vector<double> theta_alista_orig = {0.2, 0.15, 0.1};
    {
        std::ofstream f(alista_path, std::ios::binary);
        int K_a = 3;
        f.write(reinterpret_cast<const char*>(&K_a), sizeof(int));
        for (int k = 0; k < K_a; ++k) {
            f.write(reinterpret_cast<const char*>(&gamma_orig[k]), sizeof(double));
            f.write(reinterpret_cast<const char*>(&theta_alista_orig[k]), sizeof(double));
        }
    }

    auto params = unfolding::load_alista_weights(alista_path);
    assert(params.gamma.size() == 3);
    for (int k = 0; k < 3; ++k) {
        assert_near(params.gamma[k], gamma_orig[k]);
        assert_near(params.theta[k], theta_alista_orig[k]);
    }

    // Cleanup
    std::filesystem::remove(path);
    std::filesystem::remove(alista_path);

    std::cout << "All weights_loader tests passed.\n";
    return 0;
}
