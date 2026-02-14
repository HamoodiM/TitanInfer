# TitanInfer Architecture

## System Overview

TitanInfer is a layered inference engine where each layer builds on the one below. Data flows upward from raw tensors through operations, layers, and models to the thread-safe public API.

```
┌─────────────────────────────────────────┐
│           C API (titaninfer_c.h)        │  FFI boundary
├─────────────────────────────────────────┤
│         ModelHandle (model_handle.hpp)  │  Thread-safe RAII wrapper
├─────────────────────────────────────────┤
│      InferenceEngine (inference_engine) │  Buffer pre-allocation, profiling
├─────────────────────────────────────────┤
│    Model I/O (model_serializer/parser)  │  .titan binary format
├─────────────────────────────────────────┤
│        Sequential (sequential.hpp)      │  Layer container
├─────────────────────────────────────────┤
│  Layers: Dense, ReLU, Sigmoid, Tanh,   │  Forward pass implementations
│          Softmax (layer.hpp base)       │
├─────────────────────────────────────────┤
│    Ops: matmul, matvec, activations     │  Naive + AVX2 SIMD variants
├─────────────────────────────────────────┤
│          Tensor (tensor.hpp)            │  N-dim, RAII, 32-byte aligned
└─────────────────────────────────────────┘
```

## Namespace Organization

| Namespace | Purpose | Key Types |
|---|---|---|
| `titaninfer` | Core types + public API | `Tensor`, `ModelHandle`, `Logger`, exceptions |
| `titaninfer::ops` | Free-function operations | `matmul`, `relu`, `sigmoid`, `softmax` |
| `titaninfer::ops::simd` | AVX2-optimized operations | `matmul_avx2`, `cpu_supports_avx2_fma` |
| `titaninfer::layers` | Layer classes | `Layer`, `DenseLayer`, `ReluLayer`, `Sequential` |
| `titaninfer::io` | Serialization | `ModelSerializer`, `ModelParser`, `LayerType` |
| `titaninfer::engine` | Inference pipeline | `InferenceEngine`, `InferenceStats` |

## Memory Model

### RAII Aligned Allocation

All `Tensor` memory is 32-byte aligned for AVX2 compatibility. Platform-specific allocation:

| Platform | Allocate | Deallocate |
|---|---|---|
| Windows | `_aligned_malloc` | `_aligned_free` |
| macOS | `posix_memalign` | `std::free` |
| Linux | `std::aligned_alloc` | `std::free` |

The `Tensor` class follows the Rule of Five: deep-copy constructor/assignment, zero-copy move constructor/assignment (pointer transfer + nullify source), and destructor that frees aligned memory.

### Buffer Pre-Allocation

`InferenceEngine` pre-allocates one output buffer per layer at model load time by chaining `output_shape()` calls through the layer stack. This eliminates all heap allocations during inference. The buffers are reused across calls — `predict()` returns a deep copy of the final buffer.

### Zero-Copy Moves

`Tensor` move operations transfer the raw pointer without copying data. This enables efficient return-by-value from functions and container operations without performance penalty.

## SIMD Strategy

### Conditional Compilation

SIMD code is isolated behind two gates:

1. **Build-time:** CMake probes for `-mavx2 -mfma` support. If available, defines `TITANINFER_ENABLE_SIMD` and compiles `matrix_ops_simd.cpp`.
2. **Header-time:** `#if defined(TITANINFER_ENABLE_SIMD)` guards all AVX2 includes and declarations.

This ensures the project compiles on any platform (ARM, older x86) without modification.

### Blocked Matrix Multiplication

The AVX2 matmul uses cache-friendly blocking:

```
Block parameters: MC=64, NC=64, KC=256
Three-level loop: rows → K-dimension → columns
Inner kernel: 8-wide float via _mm256_loadu_ps
FMA: _mm256_fmadd_ps when __FMA__ defined, else mul+add
Scalar tail: handles non-multiple-of-8 remainders
```

### Runtime Detection

`cpu_supports_avx2_fma()` uses CPUID at runtime to verify hardware support, providing a safety check independent of compile-time flags.

## Serialization Format (.titan)

Binary format, little-endian (native x86):

```
┌──────────────────────────────────────┐
│ Magic: "TITN"           (4 bytes)    │
│ Version: 1              (uint32)     │
│ Layer count: N           (uint32)    │
├──────────────────────────────────────┤
│ Layer 0:                             │
│   Type enum              (uint32)    │
│   [Dense only]:                      │
│     in_features          (uint32)    │
│     out_features         (uint32)    │
│     has_bias             (uint8)     │
│     weights   (out×in floats)        │
│     bias      (out floats, if bias)  │
│   [Activation]: no extra data        │
├──────────────────────────────────────┤
│ Layer 1...N-1: same structure        │
└──────────────────────────────────────┘

LayerType enum: DENSE=1, RELU=2, SIGMOID=3, TANH=4, SOFTMAX=5
```

## Thread Safety

### ModelHandle Mutex

`ModelHandle` contains a `mutable std::mutex` that guards all public methods. Multiple threads can safely call `predict()`, `stats()`, and `reset_stats()` on the same handle concurrently. The mutex serializes access to the underlying `InferenceEngine`, whose internal buffers are shared mutable state.

### Logger Singleton

The `Logger` uses a function-local static (C++17 magic static guarantee for thread-safe initialization). All `log()` calls are mutex-guarded. The level check is performed before acquiring the lock to avoid contention when messages are filtered.

### InferenceEngine is Not Thread-Safe

`InferenceEngine` itself has no synchronization. Thread safety is provided by the `ModelHandle` wrapper. Direct use of `InferenceEngine` from multiple threads requires external synchronization.

## Exception Hierarchy

```
std::runtime_error
  └── TitanInferException (+ ErrorCode)
        ├── ModelLoadException     FILE_NOT_FOUND, INVALID_FORMAT, EMPTY_MODEL
        ├── InferenceException     NO_MODEL_LOADED
        └── ValidationException    SHAPE_MISMATCH, NAN_INPUT
```

Exception translation occurs only at the `ModelHandle` boundary. `InferenceEngine` and lower layers throw raw `std::runtime_error` / `std::invalid_argument`. `ModelHandle` catches these and rethrows as structured exceptions.

## API Pattern

All operations follow a consistent signature convention:

```cpp
void op(const Tensor& input, Tensor& output);
```

The output tensor is auto-allocated if its shape doesn't match the expected output. This pattern enables callers to either pre-allocate for performance or let the operation handle allocation.
