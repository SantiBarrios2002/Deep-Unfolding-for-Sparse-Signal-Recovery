#include "solvers/fista_solver.hpp"
#include "math/soft_threshold.hpp"
#include "math/matrix_utils.hpp"

namespace unfolding {

FistaSolver::FistaSolver(double lambda, int max_iter, double tol)
    : lambda_(lambda), max_iter_(max_iter), tol_(tol) {}

Eigen::VectorXd FistaSolver::solve(const Eigen::VectorXd& y,
                                     const Eigen::MatrixXd& A) {
    history_.clear();

    int n = A.cols();
    double L = compute_lipschitz_constant(A);
    double threshold = lambda_ / L;

    Eigen::MatrixXd At = A.transpose();
    Eigen::VectorXd Aty = At * y;

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd x_prev = x;
    double t = 1.0;

    for (int k = 0; k < max_iter_; ++k) {
        // Gradient step on x (with momentum point)
        Eigen::VectorXd z = x - (1.0 / L) * (At * (A * x) - Aty);

        // Proximal step
        Eigen::VectorXd x_new = soft_threshold(z, threshold);

        // Nesterov momentum update
        double t_new = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
        Eigen::VectorXd x_momentum = x_new + ((t - 1.0) / t_new) * (x_new - x_prev);

        // Check convergence
        double change = (x_new - x).norm();
        history_.push_back(change);

        x_prev = x_new;
        x = x_momentum;
        t = t_new;

        if (change < tol_) break;
    }

    return x_prev;  // Return the thresholded iterate, not the momentum point
}

}  // namespace unfolding
