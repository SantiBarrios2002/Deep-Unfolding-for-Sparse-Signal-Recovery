cmake_minimum_required(VERSION 3.18)
project(deep_unfolding VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Eigen 3.4+ ---
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally, fetching via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# --- Optional: AudioFile (header-only .wav I/O) ---
option(USE_AUDIOFILE "Enable AudioFile for .wav I/O" OFF)
if(USE_AUDIOFILE)
    add_library(audiofile INTERFACE)
    target_include_directories(audiofile INTERFACE ${CMAKE_SOURCE_DIR}/external/AudioFile)
endif()

# --- Core library ---
add_library(unfolding_core
    src/solvers/ista_solver.cpp
    src/solvers/fista_solver.cpp
    src/solvers/music_doa.cpp
    src/neural/lista_network.cpp
    src/neural/alista_network.cpp
    src/neural/weights_loader.cpp
    src/applications/compressed_sensing.cpp
    src/applications/doa_estimator.cpp
    src/applications/audio_inpainter.cpp
)
target_include_directories(unfolding_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(unfolding_core PUBLIC Eigen3::Eigen)
if(USE_AUDIOFILE)
    target_link_libraries(unfolding_core PUBLIC audiofile)
    target_compile_definitions(unfolding_core PUBLIC HAS_AUDIOFILE)
endif()

# --- Main executable ---
add_executable(deep_unfolding src/main.cpp)
target_link_libraries(deep_unfolding PRIVATE unfolding_core)

# --- Benchmark executables ---
set(BENCH_TARGETS
    bench_synthetic
    bench_doa
    bench_speed
    bench_audio
    bench_ablation
)
foreach(bench ${BENCH_TARGETS})
    add_executable(${bench} src/benchmarks/${bench}.cpp)
    target_link_libraries(${bench} PRIVATE unfolding_core)
endforeach()

# --- Tests (CTest) ---
enable_testing()

set(TEST_SOURCES
    tests/test_soft_threshold.cpp
    tests/test_ista_convergence.cpp
    tests/test_lista_matches_ista.cpp
    tests/test_steering_vector.cpp
    tests/test_weights_loader.cpp
)
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE unfolding_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
