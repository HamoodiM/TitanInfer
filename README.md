# TitanInfer

A neural-network inference library in C++ with its own tensor, matrix kernels and model format, and no BLAS, Eigen or LAPACK dependency.

Portfolio page, with an interactive model of the cache blocking: [hamoodim.github.io/projects/titaninfer](https://hamoodim.github.io/projects/titaninfer/)

## Context

- Solo project, Feb 8 to Mar 28, 2026 (29 commits).
- Built in phases: tensor and matrix ops, AVX2 kernels, layers, the `.titan` format, the inference engine, the public and C APIs, then CNN layers, quantization and fusion, then the model server and cluster controller.
- About 7,800 lines in `src/`, `include/` and `bin/`, and about 6,600 lines of tests.

## Features

**Core library**
- N-dimensional `Tensor` with RAII ownership of 32-byte-aligned memory (`_aligned_malloc`, `posix_memalign` or `std::aligned_alloc` by platform), copy and move.
- Scalar matrix ops, plus a cache-blocked AVX2 matmul (MC=64, KC=256, NC=64) that uses FMA when available. The AVX2 kernel has its own tests and benchmark, but the Dense, fused and Conv2D layers still call the scalar `ops::matmul` / `ops::matvec`.
- Layers: Dense, ReLU, Sigmoid, Tanh, Softmax, Conv2D, MaxPool2D, AvgPool2D, Flatten, in a `Sequential` container.
- `.titan` binary format (magic `TITN`, format version 2) with a writer and a parser that rejects newer versions.
- `scripts/export_to_titan.py`: exports PyTorch MLPs (`nn.Linear`, ReLU, Sigmoid, Tanh, Softmax) to `.titan`.
- `InferenceEngine`: allocates one output buffer per layer when the model loads and reuses them on every call; supports warmup runs and per-layer profiling.
- `ModelHandle`: builder-style loading, a mutex around `predict`, an exception hierarchy with error codes, and a logger.
- C API in `titaninfer_c.h`: `titaninfer_load`, `titaninfer_predict`, `titaninfer_free`, `titaninfer_last_error`, `titaninfer_is_loaded`, `titaninfer_layer_count`, `titaninfer_inference_count`, `titaninfer_mean_latency_ms`.

**Added after the original README**
- INT8 quantization: per-tensor affine `q = clamp(round(x / scale) + zero_point, -128, 127)`, an SSE int8 kernel (widen with `_mm_cvtepi8_epi16`, multiply-add with `_mm_madd_epi16`, which avoids the int16 saturation of `_mm_maddubs_epi16`) and `QuantizedDenseLayer`.
- Layer fusion: Dense+ReLU and Dense+Sigmoid become single fused layers (`engine/fusion.hpp`, `layers/fused_layers.hpp`), plus a `ModelCompiler` pass over a `Sequential` model.
- `ThreadPool`, and a batcher (`DynamicBatcher`) that groups concurrent requests.
- `ModelServer`: several models and versions behind `/v1/models/{name}[/versions/{ver}]/predict` routes, LRU model eviction and per-tenant quotas.
- `ClusterController`: routes requests with power-of-two random choices over a load score (0.4 × queue depth + 0.3 × CPU + 0.3 × p95/SLO), with a tenant-affinity tie-break, a per-node-and-model circuit breaker (closed, open, half-open) and autoscaling signals.
- Optional gRPC service and cluster binaries (`USE_GRPC=ON`), Docker images, and a GitHub Actions chaos test that runs a 3-node cluster.

## Building

Requires CMake 3.14+ and a C++20 compiler whose standard library has `std::jthread` (the root `CMakeLists.txt` sets `CMAKE_CXX_STANDARD 20`, and the cluster controller uses `std::jthread`). GoogleTest 1.14.0 and Google Benchmark 1.8.3 are downloaded by CMake when tests are enabled.
Builds with MSVC 19.44 (Visual Studio 2022) on Windows; CI builds and runs the tests on ubuntu-22.04 and windows-latest.

```bash
# Linux / macOS
bash scripts/build.sh --test          # options: --debug, --clean, --no-simd

# Windows
scripts\build.bat /test

# Manual
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest --output-on-failure
```

Docker:

```bash
docker build -t titaninfer --target builder .
docker run --rm titaninfer sh -c "cd build && ctest --output-on-failure"
docker-compose run test
```

| CMake option | Default | Effect |
|---|---|---|
| `ENABLE_SIMD` | `ON` | AVX2/FMA flags when the compiler supports them |
| `BUILD_TESTS` | `ON` | Test executables and benchmarks |
| `BUILD_EXAMPLES` | `ON` | `simple_inference`, `batch_processing`, `c_api_usage` |
| `USE_GRPC` | `OFF` | Cluster controller and model-server node binaries (needs gRPC and Protobuf) |

## Tests

346 GoogleTest cases (294 `TEST`, 52 `TEST_F`) across 20 test executables; `matrix_ops_simd_test` is only built when SIMD is available. CI runs on ubuntu-22.04 and windows-latest (`.github/workflows/ci.yml`), with separate `quality.yml` and `chaos.yml` workflows.

```bash
cd build && ctest --output-on-failure
bash scripts/run_all_tests.sh --valgrind
bash scripts/run_all_tests.sh --coverage
```

## Performance

Benchmarks live in `benchmarks/` (Google Benchmark): `matrix_benchmark`, `inference_benchmark`, `conv_benchmark`, `thread_pool_benchmark`.

```bash
./tests/matrix_benchmark --benchmark_repetitions=5
```

Results on an Intel Core i5-14400F (Windows, MSVC Release with `/arch:AVX2`, median of 3 repetitions, Nov 2026):

| Size | Naive | AVX2 kernel | Speed-up |
|---|---|---|---|
| 64 × 64 | 70.6 µs | 62.5 µs | 1.13× |
| 128 × 128 | 0.90 ms | 0.73 ms | 1.23× |
| 256 × 256 | 8.92 ms | 8.40 ms | 1.06× |
| 512 × 512 | 69.6 ms | 69.1 ms | 1.01× |

The kernel is only slightly faster, and the gap closes as matrices grow. The likely cause is the strided gather of B (see the design notes); packing B is the next change to measure. Raw output: `docs/benchmark_2026-11.txt`.

## Design notes

- **Pre-allocated layer outputs.** The engine walks the model once at load time, computes each layer's output shape and allocates its output buffer, so layer outputs reuse the same memory on every call. Inference is not allocation-free: the Dense layer's batched (2-D) path allocates a transposed copy of its weights on every call, and `predict()` returns a copy of the last buffer.
- **Coarse locking.** `ModelHandle` serializes `predict` calls with one mutex. That keeps the engine's shared buffers safe, at the cost of running concurrent calls on one handle one at a time. `DynamicBatcher` groups concurrent requests into batches.
- **Gathering B in the AVX2 kernel.** B is row-major, so the kernel builds each 8-wide column vector with `_mm256_set_ps` from 8 strided loads. Packing B into a transposed block buffer is the next optimization for the kernel itself.
- **Format versioning.** The parser accepts any version up to the one it knows and rejects newer files with a clear error instead of misreading them.

## Known issues and next steps

- The layers don't use the AVX2 kernel yet. Wiring Dense, fused Dense and Conv2D to it is the first next step, followed by measuring the end-to-end difference.
- The Dense layer's 2-D path allocates and transposes its weights on every call; storing them pre-transposed at load time would remove that.
- `predict()` copies its output; an overload that writes into a caller-owned tensor would avoid it.
- The PyTorch exporter writes format version 1 (the C++ writer uses 2) and can't export Conv2D or pooling layers.
- The AVX2 kernel is only 1.0–1.2× faster than the naive loop (see Benchmarks); packing B should come before wiring it into the layers.
- `docs/PERFORMANCE.md` lists approximate speedups that were never measured; the table above replaces them.

## How it was built

Designed and built by Mohammad Mohammad. The final cleanup commits were made with an AI assistant and are marked as co-authored.
