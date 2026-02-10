#pragma once

#include "neural/lista_network.hpp"
#include "neural/alista_network.hpp"
#include <string>
#include <vector>

namespace unfolding {

/// Load LISTA weights from binary file exported by Python
/// Format: [K, n, m] (int32) then K × [S(n×n), W_e(n×m), theta(n)] (float64)
ListaNetwork load_lista_weights(const std::string& filepath);

/// Load ALISTA weights from binary file
/// Format: [K] (int32) then K × [gamma, theta] (float64)
/// The sensing matrix A must be provided separately.
struct AlistaParams {
    std::vector<double> gamma;
    std::vector<double> theta;
};

AlistaParams load_alista_weights(const std::string& filepath);

}  // namespace unfolding
