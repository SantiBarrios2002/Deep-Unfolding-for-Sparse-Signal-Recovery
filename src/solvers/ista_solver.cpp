#include "solvers/ista_solver.hpp"
#include "math/soft_threshold.hpp"
#include "math/matrix_utils.hpp"

namespace unfolding {

IstaSolver::IstaSolver(double lambda, int max_iter, double tol)
    : lambda_(lambda), max_iter_(max_iter), tol_(tol) {}

Eigen::VectorXd IstaSolver::solve(const Eigen::VectorXd& y,
                                    const Eigen::MatrixXd& A) {
    history_.clear();

    int n = A.cols();
    double L = compute_lipschitz_constant(A);
    double threshold = lambda_ / L;

    // Precompute A^T and A^T * y
    Eigen::MatrixXd At = A.transpose();
    Eigen::VectorXd Aty = At * y;

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    for (int k = 0; k < max_iter_; ++k) {
        // Gradient step: z = x - (1/L) * A^T(Ax - y)
        Eigen::VectorXd z = x - (1.0 / L) * (At * (A * x) - Aty);

        // Proximal step: soft-thresholding
        Eigen::VectorXd x_new = soft_threshold(z, threshold);

        // Check convergence
        double change = (x_new - x).norm();
        x = x_new;
        history_.push_back(change);

        if (change < tol_) break;
    }

    return x;
}

}  // namespace unfolding
