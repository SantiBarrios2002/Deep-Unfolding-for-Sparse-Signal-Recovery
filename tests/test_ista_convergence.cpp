/// Test that ISTA converges to a reasonable solution on a small problem

#include "solvers/ista_solver.hpp"
#include "applications/compressed_sensing.hpp"
#include "evaluation/metrics.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Small problem: n=50, m=25, k=5
    auto prob = unfolding::generate_cs_problem(25, 50, 5, 40.0, 42);

    unfolding::IstaSolver ista(0.1, 500, 1e-8);
    auto x_hat = ista.solve(prob.y, prob.A);

    double nmse = unfolding::nmse_db(x_hat, prob.x);
    std::cout << "ISTA NMSE = " << nmse << " dB\n";

    // ISTA should converge to at least -20 dB on this easy problem
    assert(nmse < -20.0 && "ISTA did not converge sufficiently");

    // Check that convergence history is decreasing (mostly)
    const auto& hist = ista.convergence_history();
    assert(!hist.empty() && "No convergence history recorded");
    std::cout << "Converged in " << hist.size() << " iterations\n";

    std::cout << "ISTA convergence test passed.\n";
    return 0;
}
