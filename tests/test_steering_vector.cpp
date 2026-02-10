/// Test steering vector phase correctness

#include "applications/doa_estimator.hpp"
#include <iostream>
#include <cmath>
#include <complex>
#include <cassert>

void assert_near(double a, double b, double tol = 1e-10) {
    if (std::abs(a - b) > tol) {
        std::cerr << "FAIL: " << a << " != " << b << "\n";
        assert(false);
    }
}

int main() {
    int M = 4;
    double d_lambda = 0.5;

    // Test at broadside (theta = 0°): all phases should be 0
    Eigen::VectorXd theta_broadside(1);
    theta_broadside << 0.0;
    auto D = unfolding::build_steering_matrix(M, d_lambda, theta_broadside);

    for (int i = 0; i < M; ++i) {
        assert_near(D(i, 0).real(), 1.0);
        assert_near(D(i, 0).imag(), 0.0);
    }

    // Test at endfire (theta = 90°): phase increment = 2π * d/λ * sin(90°) = π
    Eigen::VectorXd theta_endfire(1);
    theta_endfire << M_PI / 2.0;  // 90°
    D = unfolding::build_steering_matrix(M, d_lambda, theta_endfire);

    // Phase of element i should be i*π
    for (int i = 0; i < M; ++i) {
        double expected_phase = i * M_PI;
        double actual_real = std::cos(expected_phase);
        double actual_imag = std::sin(expected_phase);
        assert_near(D(i, 0).real(), actual_real, 1e-10);
        assert_near(D(i, 0).imag(), actual_imag, 1e-10);
    }

    // Test that steering vectors have unit norm (||a(θ)||₂ = √M)
    Eigen::VectorXd theta_grid(5);
    theta_grid << -M_PI / 3, -M_PI / 6, 0, M_PI / 6, M_PI / 3;
    D = unfolding::build_steering_matrix(M, d_lambda, theta_grid);

    for (int j = 0; j < theta_grid.size(); ++j) {
        double col_norm = D.col(j).norm();
        assert_near(col_norm, std::sqrt(M));
    }

    std::cout << "All steering vector tests passed.\n";
    return 0;
}
