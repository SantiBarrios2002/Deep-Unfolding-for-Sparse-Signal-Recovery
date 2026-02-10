#include <iostream>
#include <string>

#include "utils/config.hpp"

void print_usage() {
    std::cout << "Usage: deep_unfolding <command>\n\n"
              << "Commands:\n"
              << "  cs       Run synthetic compressed sensing experiment\n"
              << "  doa      Run DOA estimation experiment\n"
              << "  speed    Run speed-accuracy benchmark\n"
              << "  audio    Run audio inpainting demo\n"
              << "  ablation Run generalization & ablation studies\n"
              << "  help     Show this help message\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    unfolding::Config config;

    if (cmd == "cs") {
        std::cout << "Running synthetic compressed sensing experiment...\n";
        // TODO: call bench_synthetic logic
        std::cout << "Not yet implemented.\n";
    } else if (cmd == "doa") {
        std::cout << "Running DOA estimation experiment...\n";
        // TODO: call bench_doa logic
        std::cout << "Not yet implemented.\n";
    } else if (cmd == "speed") {
        std::cout << "Running speed-accuracy benchmark...\n";
        // TODO: call bench_speed logic
        std::cout << "Not yet implemented.\n";
    } else if (cmd == "audio") {
        std::cout << "Running audio inpainting demo...\n";
        // TODO: call bench_audio logic
        std::cout << "Not yet implemented.\n";
    } else if (cmd == "ablation") {
        std::cout << "Running ablation studies...\n";
        // TODO: call bench_ablation logic
        std::cout << "Not yet implemented.\n";
    } else {
        std::cerr << "Unknown command: " << cmd << "\n";
        print_usage();
        return 1;
    }

    return 0;
}
