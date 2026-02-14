# ============================================================
# TitanInfer Test Targets
# ============================================================

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

include(GoogleTest)

# Helper function to reduce boilerplate
function(titaninfer_add_test NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE titaninfer GTest::gtest_main)
    gtest_discover_tests(${NAME})
endfunction()

# Core tests
titaninfer_add_test(tensor_test          tensor_test.cpp)
titaninfer_add_test(matrix_ops_test      ops/matrix_ops_test.cpp)
titaninfer_add_test(activations_test     ops/activations_test.cpp)
titaninfer_add_test(layer_test           layers/layer_test.cpp)
titaninfer_add_test(sequential_test      layers/sequential_test.cpp)
titaninfer_add_test(serialization_test   io/serialization_test.cpp)
titaninfer_add_test(inference_engine_test engine/inference_engine_test.cpp)
titaninfer_add_test(api_test             api/test_api.cpp)

# SIMD-only test and benchmark
if(SIMD_AVAILABLE)
    titaninfer_add_test(matrix_ops_simd_test ops/matrix_ops_simd_test.cpp)

    add_executable(matrix_benchmark ../benchmarks/matrix_benchmark.cpp)
    target_link_libraries(matrix_benchmark
        PRIVATE
        titaninfer
        benchmark::benchmark
    )
endif()
