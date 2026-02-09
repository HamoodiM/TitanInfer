# TitanInfer: Zero-Dependency C++ Deep Learning Inference Engine

High-performance deep learning inference engine built from scratch in C++17 with manual implementations of all linear algebra operations. No external dependencies on Eigen, OpenBLAS, or LAPACK.

## Features

- N-dimensional tensor support with dynamic shapes
- 32-byte aligned memory for AVX2 SIMD operations
- RAII-based memory management with copy and move semantics
- Matrix operations: multiplication, transpose, element-wise operations
- Cross-platform (Linux, macOS, Windows via Docker)

## Dependencies

- Docker 20.10+
- C++17 compiler (GCC 11+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- GoogleTest 1.14.0 (auto-downloaded)

## Build

```bash
docker build -t titaninfer .

# All tests
docker run --rm titaninfer

# Specific test suites
docker run --rm titaninfer sh -c "cd build && ./tensor_test"
docker run --rm titaninfer sh -c "cd build && ./matrix_ops_test"

# Interactive container
docker run --rm -it titaninfer /bin/bash
```