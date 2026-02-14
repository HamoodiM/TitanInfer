cmake_minimum_required(VERSION 3.14)
project(TitanInfer VERSION 0.1.0 LANGUAGES C CXX)

# ============================================================
# Compiler Requirements
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Compiler Flags
# ============================================================

# Option to enable/disable SIMD flags. Useful when cross-building for
# non-x86 targets (e.g. arm64) or when compiling with toolchains that do
# not support AVX2/FMA.
option(ENABLE_SIMD "Enable SIMD compiler flags (AVX2/FMA). Turn off for non-x86 targets." ON)

# Track whether SIMD (AVX2/FMA) is actually available for this build
set(SIMD_AVAILABLE OFF)

if(MSVC)
    add_compile_options(/W4 /WX)
    if(ENABLE_SIMD)
        # MSVC uses /arch:AVX2 to enable AVX2 instructions
    add_compile_options(/arch:AVX2)
    add_compile_definitions(TITANINFER_ENABLE_SIMD)
    set(SIMD_AVAILABLE ON)
    endif()
    add_compile_definitions($<$<CONFIG:Debug>:TITANINFER_DEBUG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    if(ENABLE_SIMD)
        include(CheckCXXCompilerFlag)
        # Check whether the compiler accepts the x86-specific flags before adding them
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_MAVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_MAVX2 AND COMPILER_SUPPORTS_FMA)
            add_compile_options(-mavx2 -mfma -march=native)
            add_compile_definitions(TITANINFER_ENABLE_SIMD)
            set(SIMD_AVAILABLE ON)
        else()
            message(STATUS "SIMD flags (-mavx2 -mfma) not supported by the compiler; building without them.")
        endif()
    endif()
    add_compile_definitions($<$<CONFIG:Debug>:TITANINFER_DEBUG>)
endif()

# ============================================================
# Library Target
# ============================================================
add_library(titaninfer
    src/tensor.cpp
    src/ops/matrix_ops.cpp
    src/ops/activations.cpp
    src/layers/activation_layer.cpp
    src/layers/dense_layer.cpp
    src/layers/sequential.cpp
    src/io/model_serializer.cpp
    src/io/model_parser.cpp
    src/engine/inference_engine.cpp
    src/logger.cpp
    src/model_handle.cpp
    src/titaninfer_c.cpp
)

# Add SIMD source only if available
if(SIMD_AVAILABLE)
    target_sources(titaninfer PRIVATE src/ops/matrix_ops_simd.cpp)
endif()

target_include_directories(titaninfer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Thread support (pthread on Linux, no-op on Windows)
find_package(Threads REQUIRED)
target_link_libraries(titaninfer PUBLIC Threads::Threads)

# ============================================================
# Testing
# ============================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Fetch Google Benchmark
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
    
    # Test executables
    add_executable(tensor_test tests/tensor_test.cpp)
    add_executable(matrix_ops_test tests/ops/matrix_ops_test.cpp)

    target_link_libraries(tensor_test PRIVATE titaninfer GTest::gtest_main)
    target_link_libraries(matrix_ops_test PRIVATE titaninfer GTest::gtest_main)

    add_executable(activations_test tests/ops/activations_test.cpp)
    add_executable(layer_test tests/layers/layer_test.cpp)

    target_link_libraries(activations_test PRIVATE titaninfer GTest::gtest_main)
    target_link_libraries(layer_test PRIVATE titaninfer GTest::gtest_main)

    add_executable(sequential_test tests/layers/sequential_test.cpp)
    target_link_libraries(sequential_test PRIVATE titaninfer GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(tensor_test)
    gtest_discover_tests(matrix_ops_test)
    gtest_discover_tests(activations_test)
    gtest_discover_tests(layer_test)
    gtest_discover_tests(sequential_test)

    add_executable(serialization_test tests/io/serialization_test.cpp)
    target_link_libraries(serialization_test PRIVATE titaninfer GTest::gtest_main)
    gtest_discover_tests(serialization_test)

    add_executable(inference_engine_test tests/engine/inference_engine_test.cpp)
    target_link_libraries(inference_engine_test PRIVATE titaninfer GTest::gtest_main)
    gtest_discover_tests(inference_engine_test)

    add_executable(api_test tests/api/test_api.cpp)
    target_link_libraries(api_test PRIVATE titaninfer GTest::gtest_main)
    gtest_discover_tests(api_test)

    # SIMD-only test/benchmark: add only if SIMD is available
    if(SIMD_AVAILABLE)
        add_executable(matrix_ops_simd_test tests/ops/matrix_ops_simd_test.cpp)
        target_link_libraries(matrix_ops_simd_test PRIVATE titaninfer GTest::gtest_main)
        gtest_discover_tests(matrix_ops_simd_test)

        # Benchmark executable
        add_executable(matrix_benchmark benchmarks/matrix_benchmark.cpp)
        target_link_libraries(matrix_benchmark 
            PRIVATE 
            titaninfer 
            benchmark::benchmark
        )
    endif()
endif()

# ============================================================
# Examples
# ============================================================
option(BUILD_EXAMPLES "Build example executables" ON)

if(BUILD_EXAMPLES)
    add_executable(simple_inference examples/simple_inference.cpp)
    target_link_libraries(simple_inference PRIVATE titaninfer)

    add_executable(batch_processing examples/batch_processing.cpp)
    target_link_libraries(batch_processing PRIVATE titaninfer)

    add_executable(c_api_usage examples/c_api_usage.c)
    target_link_libraries(c_api_usage PRIVATE titaninfer)
endif()

# ============================================================
# Installation
# ============================================================
install(TARGETS titaninfer
    EXPORT TitanInferTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/titaninfer
    DESTINATION include
)
