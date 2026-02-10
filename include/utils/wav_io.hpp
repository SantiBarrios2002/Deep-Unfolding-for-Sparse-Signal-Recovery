#pragma once

#include <Eigen/Dense>
#include <string>

namespace unfolding {

/// Read a mono WAV file into an Eigen vector (samples normalized to [-1, 1])
/// Returns the sample rate via the output parameter.
Eigen::VectorXd read_wav(const std::string& filepath, int& sample_rate);

/// Write an Eigen vector as a mono WAV file
void write_wav(const std::string& filepath, const Eigen::VectorXd& audio,
               int sample_rate = 16000);

}  // namespace unfolding
