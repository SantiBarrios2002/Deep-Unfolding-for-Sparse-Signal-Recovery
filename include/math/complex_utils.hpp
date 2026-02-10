#pragma once

#include <Eigen/Dense>
#include <complex>

namespace unfolding {

/// Stack complex vector into real: [Re(y); Im(y)] ∈ ℝ^{2M}
inline Eigen::VectorXd complex_to_real_stacked(const Eigen::VectorXcd& y) {
    int M = y.size();
    Eigen::VectorXd y_real(2 * M);
    y_real.head(M) = y.real();
    y_real.tail(M) = y.imag();
    return y_real;
}

/// Stack complex matrix into real block form:
/// [Re(D), -Im(D); Im(D), Re(D)] ∈ ℝ^{2M × 2N}
inline Eigen::MatrixXd complex_to_real_stacked(const Eigen::MatrixXcd& D) {
    int M = D.rows();
    int N = D.cols();
    Eigen::MatrixXd A_real(2 * M, 2 * N);

    A_real.topLeftCorner(M, N)     =  D.real();
    A_real.topRightCorner(M, N)    = -D.imag();
    A_real.bottomLeftCorner(M, N)  =  D.imag();
    A_real.bottomRightCorner(M, N) =  D.real();

    return A_real;
}

/// Recover complex vector from stacked real: y_real = [Re; Im] → complex
inline Eigen::VectorXcd real_stacked_to_complex(const Eigen::VectorXd& y_real) {
    int M = y_real.size() / 2;
    return y_real.head(M).cast<std::complex<double>>()
         + std::complex<double>(0, 1) * y_real.tail(M).cast<std::complex<double>>();
}

}  // namespace unfolding
