#pragma once

#include <chrono>
#include <string>
#include <iostream>

namespace unfolding {

/// RAII timer using std::chrono::high_resolution_clock
class Timer {
public:
    explicit Timer(const std::string& label = "")
        : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        if (!stopped_) stop();
    }

    /// Stop and return elapsed time in microseconds
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        elapsed_us_ = static_cast<double>(us);
        stopped_ = true;
        return elapsed_us_;
    }

    double elapsed_us() const { return elapsed_us_; }

    /// Print elapsed time with label
    void print() const {
        std::cout << label_ << ": " << elapsed_us_ << " us" << std::endl;
    }

private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
    double elapsed_us_ = 0.0;
    bool stopped_ = false;
};

}  // namespace unfolding
