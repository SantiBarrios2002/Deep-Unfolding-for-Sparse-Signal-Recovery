#pragma once

#include <Eigen/Dense>
#include <vector>

namespace unfolding {

class AudioInpainter {
public:
    /// frame_size = DCT window length, overlap = overlap between frames
    AudioInpainter(int frame_size = 512, int overlap = 256);

    /// Randomly corrupt audio by zeroing a fraction of samples
    Eigen::VectorXd corrupt(const Eigen::VectorXd& audio, double drop_fraction,
                             Eigen::VectorXi& mask, unsigned int seed = 42) const;

    /// Recover corrupted audio using a given solver (frame-wise DCT + sparse recovery)
    Eigen::VectorXd recover(const Eigen::VectorXd& corrupted,
                             const Eigen::VectorXi& mask,
                             class SolverBase& solver) const;

    /// Build the DCT dictionary matrix for a given frame size
    static Eigen::MatrixXd build_dct_matrix(int frame_size);

private:
    int frame_size_;
    int overlap_;
};

}  // namespace unfolding
