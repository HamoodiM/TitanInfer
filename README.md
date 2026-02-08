# TitanInfer: Zero-Dependency C++ Deep Learning Inference Engine

High-performance deep learning inference engine built from scratch in C++17 with manual implementations of all linear algebra operations. No external dependencies on Eigen, OpenBLAS, or LAPACK.

## Features

- N-dimensional tensor support with dynamic shapes
- 32-byte aligned memory for AVX2 SIMD operations
- RAII-based memory management
- Copy and move semantics
- Cross-platform (Linux, macOS, Windows via Docker)

## Dependencies

- Docker 20.10+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- GoogleTest 1.14.0 (auto-downloaded)

## Quick Start

### Build and Run Tests

```bash
# Build Docker image
docker build -t titaninfer:latest .

# Run test suite
docker run --rm titaninfer:latest /bin/bash -c "cd build && ctest --output-on-failure"
