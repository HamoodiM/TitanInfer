# TitanInfer: Zero-Dependency C++ Deep Learning Inference Engine

High-performance deep learning inference engine built from scratch in C++17 with manual implementations of all linear algebra operations optimized with AVX2 SIMD intrinsics. No external dependencies on Eigen, OpenBLAS, or LAPACK.

## Features

- **N-dimensional tensor support** with dynamic shapes and RAII memory management
- **32-byte aligned memory** for optimal AVX2 SIMD operations
- **Hardware-accelerated matrix operations** using AVX2 + FMA intrinsics
- **Cache-optimized algorithms** with blocking/tiling strategies
- **Comprehensive test suite** with GoogleTest (38 unit tests)
- **Performance benchmarking** using Google Benchmark framework
- **Cross-platform** (Linux, macOS, Windows via Docker)

## Performance

- **3-5x speedup** on matrix multiplication vs naive implementation
- AVX2 vectorization with 8-wide float operations
- FMA (fused multiply-add) support for maximum throughput
- Cache-friendly blocking for large matrices

## Dependencies

- Docker 20.10+
- C++17 compiler with AVX2 support (GCC 11+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- GoogleTest 1.14.0 (auto-downloaded)
- Google Benchmark 1.8.3 (auto-downloaded)

## Quick Start

### Build

```bash
docker build -t titaninfer .

# All unit tests (38 tests)
docker run --rm titaninfer sh -c "cd build && ctest --output-on-failure"

# Specific test suites
docker run --rm titaninfer sh -c "cd build && ./tensor_test"
docker run --rm titaninfer sh -c "cd build && ./matrix_ops_test"
docker run --rm titaninfer sh -c "cd build && ./matrix_ops_simd_test"

# Compare naive vs AVX2 implementations
docker run --rm titaninfer sh -c "cd build && ./matrix_benchmark --benchmark_filter='256'"

# Full benchmark suite with repetitions
docker run --rm titaninfer sh -c "cd build && ./matrix_benchmark --benchmark_repetitions=5"

# CPU Feature Detection
docker run --rm titaninfer sh -c "grep -E 'model name|avx2|fma' /proc/cpuinfo | head -3"
```

## Project Structure
```plaintext
TitanInfer/
├── CMakeLists.txt
├── Dockerfile
├── README.md
├── include/titaninfer/
│ ├── tensor.hpp
│ └── ops/
│ ├── matrix_ops.hpp
│ └── matrix_ops_simd.hpp
├── src/
│ ├── tensor.cpp
│ └── ops/
│ ├── matrix_ops.cpp
│ └── matrix_ops_simd.cpp
├── tests/
│ ├── tensor_test.cpp
│ └── ops/
│ ├── matrix_ops_test.cpp
│ └── matrix_ops_simd_test.cpp
└── benchmarks/
└── matrix_benchmark.cpp
