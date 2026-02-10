/// Test soft-thresholding operator correctness

#include "math/soft_threshold.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

void assert_near(double a, double b, double tol = 1e-12) {
    if (std::abs(a - b) > tol) {
        std::cerr << "FAIL: " << a << " != " << b
                  << " (diff=" << std::abs(a - b) << ")\n";
        assert(false);
    }
}

int main() {
    using namespace unfolding;

    // Scalar tests
    assert_near(soft_threshold(0.5, 0.3), 0.2);
    assert_near(soft_threshold(-0.5, 0.3), -0.2);
    assert_near(soft_threshold(0.2, 0.3), 0.0);   // Below threshold
    assert_near(soft_threshold(-0.2, 0.3), 0.0);   // Below threshold
    assert_near(soft_threshold(0.0, 0.3), 0.0);    // Zero input
    assert_near(soft_threshold(1.0, 0.0), 1.0);    // Zero threshold

    // Vector test with uniform threshold
    Eigen::VectorXd u(4);
    u << 0.5, -0.5, 0.2, -0.2;
    Eigen::VectorXd result = soft_threshold(u, 0.3);
    assert_near(result(0), 0.2);
    assert_near(result(1), -0.2);
    assert_near(result(2), 0.0);
    assert_near(result(3), 0.0);

    // Vector test with per-entry thresholds
    Eigen::VectorXd alpha(4);
    alpha << 0.1, 0.6, 0.2, 0.1;
    Eigen::VectorXd result2 = soft_threshold(u, alpha);
    assert_near(result2(0), 0.4);    // 0.5 - 0.1
    assert_near(result2(1), 0.0);    // |-0.5| < 0.6
    assert_near(result2(2), 0.0);    // |0.2| = 0.2, threshold
    assert_near(result2(3), -0.1);   // -0.2 + 0.1

    std::cout << "All soft_threshold tests passed.\n";
    return 0;
}
