# TitanInfer

Zero-dependency C++17 deep learning inference engine with manual implementations of all linear algebra operations, optimized with AVX2 SIMD intrinsics. No Eigen, OpenBLAS, or LAPACK — everything is built from scratch.

## Features

- **N-dimensional Tensor** — RAII memory management with 32-byte aligned allocation for AVX2
- **AVX2 SIMD Matrix Operations** — blocked matmul with FMA, 3-5x speedup over naive
- **Neural Network Layers** — Dense, ReLU, Sigmoid, Tanh, Softmax with auto-allocating output
- **Sequential Model** — composable layer container with ping-pong buffer execution
- **Binary Serialization** — `.titan` format for model save/load with PyTorch export script
- **Inference Engine** — pre-allocated buffers (zero heap allocs per inference), profiling, warmup
- **Thread-Safe Public API** — `ModelHandle` with mutex guards, structured exceptions, logging
- **C FFI** — opaque handle API for Python ctypes, Rust bindgen, and other FFI consumers
- **161 Unit Tests** — comprehensive GoogleTest suite across 9 test executables

## Quick Start

### Docker (recommended)

```bash
# Build and run all tests
docker build -t titaninfer --target builder .
docker run --rm titaninfer sh -c "cd build && ctest --output-on-failure"

# Or use docker-compose
docker-compose run test
docker-compose run benchmark
```

### Native Build

```bash
# Linux / macOS
bash scripts/build.sh --test

# Windows
scripts\build.bat /test
```

### CMake (manual)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

## API Usage

### C++ (ModelHandle)

```cpp
#include "titaninfer/TitanInfer.hpp"

// Load model
auto model = titaninfer::ModelHandle::Builder()
    .setModelPath("model.titan")
    .enableProfiling()
    .setWarmupRuns(3)
    .build();

// Run inference (thread-safe)
titaninfer::Tensor input({4});
input.fill(1.0f);
titaninfer::Tensor output = model.predict(input);

// Check profiling stats
auto stats = model.stats();
std::cout << "Latency: " << stats.mean_latency_ms << " ms\n";
```

### C API

```c
#include "titaninfer/titaninfer_c.h"

size_t shape[] = {4};
TitanInferModelHandle model = titaninfer_load("model.titan", shape, 1);

float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
float output[3];
size_t output_len;
titaninfer_predict(model, input, 4, output, 3, &output_len);

titaninfer_free(model);
```

## Architecture

```
Tensor  ──>  Ops (matmul, activations)
               │
               v
           Layers (Dense, ReLU, Sigmoid, Tanh, Softmax)
               │
               v
           Sequential (layer container)
               │
               v
         InferenceEngine (buffer pre-allocation, profiling)
               │
               v
          ModelHandle (thread-safe RAII wrapper, exception translation)
               │
               v
           C API (opaque handle FFI)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Project Structure

```
TitanInfer/
├── CMakeLists.txt                    Root build configuration
├── Dockerfile                        Multi-stage Docker build
├── docker-compose.yml                Container orchestration
├── include/titaninfer/
│   ├── TitanInfer.hpp                Single-include public API
│   ├── tensor.hpp                    N-dimensional tensor (RAII, aligned)
│   ├── exceptions.hpp                Exception hierarchy + error codes
│   ├── logger.hpp                    Thread-safe configurable logger
│   ├── model_handle.hpp              Thread-safe model wrapper
│   ├── titaninfer_c.h                C-compatible FFI header
│   ├── ops/
│   │   ├── matrix_ops.hpp            Naive matrix operations
│   │   ├── matrix_ops_simd.hpp       AVX2-optimized operations
│   │   └── activations.hpp           Activation functions
│   ├── layers/
│   │   ├── layer.hpp                 Abstract layer base class
│   │   ├── dense_layer.hpp           Fully connected layer
│   │   ├── activation_layer.hpp      Activation layers
│   │   └── sequential.hpp            Sequential model container
│   ├── io/
│   │   ├── format.hpp                .titan binary format spec
│   │   ├── model_serializer.hpp      Model save
│   │   └── model_parser.hpp          Model load
│   └── engine/
│       └── inference_engine.hpp      High-performance inference
├── src/                              Implementation files
│   └── CMakeLists.txt                Library target
├── tests/                            GoogleTest suites (161 tests)
│   ├── CMakeLists.txt                Test targets
│   ├── tensor_test.cpp
│   ├── ops/                          Matrix ops + activations tests
│   ├── layers/                       Layer + sequential tests
│   ├── io/                           Serialization tests
│   ├── engine/                       Inference engine tests
│   └── api/                          Public API + thread safety tests
├── examples/                         Integration examples
│   ├── CMakeLists.txt
│   ├── simple_inference.cpp          Minimal C++ usage
│   ├── batch_processing.cpp          Multi-threaded inference
│   └── c_api_usage.c                 Pure C integration
├── benchmarks/
│   └── matrix_benchmark.cpp          Naive vs AVX2 benchmarks
├── scripts/
│   ├── build.sh                      Linux/macOS build script
│   ├── build.bat                     Windows build script
│   ├── run_all_tests.sh              Test runner (Valgrind, coverage)
│   └── export_to_titan.py            PyTorch model export
├── docs/
│   ├── ARCHITECTURE.md               System design
│   └── PERFORMANCE.md                Optimization guidelines
└── .github/workflows/
    └── ci.yml                        CI/CD pipeline
```

## Build Options

| CMake Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | — | `Release` (optimized) or `Debug` (bounds checking) |
| `ENABLE_SIMD` | `ON` | AVX2/FMA compiler flags; auto-detected |
| `BUILD_TESTS` | `ON` | Build test and benchmark executables |
| `BUILD_EXAMPLES` | `ON` | Build example programs |

## Testing

```bash
# All 161 tests
ctest --output-on-failure

# Individual suites
./tests/tensor_test              # 20 tests: alignment, copy/move, indexing
./tests/matrix_ops_test          # 19 tests: matmul, transpose, element-wise
./tests/activations_test         # 18 tests: relu, sigmoid, tanh, softmax
./tests/layer_test               #  6 tests: polymorphic forward
./tests/sequential_test          # 23 tests: DenseLayer + Sequential
./tests/serialization_test       # 20 tests: round-trip, error handling, format
./tests/inference_engine_test    # 20 tests: predict, profiling, validation
./tests/api_test                 # 36 tests: exceptions, logger, ModelHandle,
                                 #           thread safety, C API, resources
./tests/matrix_ops_simd_test     #  5 tests: AVX2 correctness (SIMD builds only)

# With Valgrind
bash scripts/run_all_tests.sh --valgrind

# With coverage
bash scripts/run_all_tests.sh --coverage
```

## Performance

- **3-5x speedup** on matrix multiplication with AVX2 vs naive implementation
- **Zero heap allocations** per inference call (buffers pre-allocated at load time)
- **FMA support** for fused multiply-add throughput
- **Cache-friendly blocking** with tuned MC=64, NC=64, KC=256 parameters

Run benchmarks:
```bash
./tests/matrix_benchmark --benchmark_filter='(Naive_256|AVX2_256)'
./tests/matrix_benchmark --benchmark_repetitions=5
```

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for optimization details.

## Dependencies

- **C++17 compiler**: GCC 11+, Clang 10+, MSVC 2019+
- **CMake 3.14+**
- **GoogleTest 1.14.0** (auto-downloaded via FetchContent)
- **Google Benchmark 1.8.3** (auto-downloaded via FetchContent)
- **Docker 20.10+** (optional, for containerized builds)
- **AVX2 + FMA CPU** (Intel Haswell / AMD Excavator or newer; graceful fallback without)

## License

See LICENSE file.
