#pragma once

#include <Eigen/Dense>
#include <vector>

namespace unfolding {

class MusicDoa {
public:
    /// M = number of antennas, num_sources = known number of sources
    MusicDoa(int M, int num_sources, double d_lambda = 0.5);

    /// Compute the MUSIC pseudo-spectrum over a theta grid (radians)
    Eigen::VectorXd compute_spectrum(const Eigen::MatrixXcd& R,
                                      const Eigen::VectorXd& theta_grid) const;

    /// Estimate DOAs from a covariance matrix R
    std::vector<double> estimate_doas(const Eigen::MatrixXcd& R,
                                       const Eigen::VectorXd& theta_grid) const;

private:
    int M_;
    int num_sources_;
    double d_lambda_;
};

}  // namespace unfolding
