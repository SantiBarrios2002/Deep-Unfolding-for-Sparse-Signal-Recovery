/// Test that LISTA initialized with ISTA parameters reproduces ISTA output

#include "solvers/ista_solver.hpp"
#include "neural/lista_network.hpp"
#include "applications/compressed_sensing.hpp"
#include "evaluation/metrics.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    // Small problem for fast testing
    auto prob = unfolding::generate_cs_problem(25, 50, 5, 40.0, 42);

    int K = 16;
    double lambda = 0.1;

    // Run ISTA for exactly K iterations
    unfolding::IstaSolver ista(lambda, K, 0.0);  // tol=0 to force all K iterations
    auto x_ista = ista.solve(prob.y, prob.A);

    // Run LISTA initialized with ISTA parameters (K layers)
    auto lista = unfolding::ListaNetwork::from_ista_params(prob.A, lambda, K);
    auto x_lista = lista.solve(prob.y, prob.A);

    // They should produce the same output (up to floating-point precision)
    double diff = (x_ista - x_lista).norm();
    double rel_diff = diff / std::max(x_ista.norm(), 1e-15);

    std::cout << "||x_ista - x_lista|| = " << diff << "\n";
    std::cout << "Relative difference  = " << rel_diff << "\n";

    assert(rel_diff < 1e-10 && "LISTA with ISTA params should match ISTA output");

    std::cout << "LISTA-matches-ISTA test passed.\n";
    return 0;
}
