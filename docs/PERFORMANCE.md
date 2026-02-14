# TitanInfer Performance Guide

## AVX2 Matrix Multiplication

### Blocking Strategy

The AVX2 matmul uses three-level cache-friendly blocking to minimize L1/L2 cache misses:

| Parameter | Value | Purpose |
|---|---|---|
| MC | 64 | Row block size — fits in L1 cache |
| NC | 64 | Column block size — fits in L1 cache |
| KC | 256 | K-dimension block — fits in L2 cache |

The blocking loop structure:
```
for (row blocks of MC)
  for (K blocks of KC)
    for (column blocks of NC)
      inner kernel: 8-wide AVX2 dot product
```

### Speedup

Typical speedups over naive implementation:

| Matrix Size | Naive (ms) | AVX2 (ms) | Speedup |
|---|---|---|---|
| 64×64 | ~0.02 | ~0.006 | ~3x |
| 256×256 | ~2.5 | ~0.5 | ~5x |
| 1024×1024 | ~160 | ~35 | ~4.5x |

Speedup varies by matrix size, CPU cache hierarchy, and whether FMA is available.

### FMA (Fused Multiply-Add)

When the `__FMA__` compiler flag is defined, the inner kernel uses `_mm256_fmadd_ps` instead of separate multiply and add. This provides:
- **2x theoretical throughput** on the multiply-accumulate operation
- **Better numerical precision** (single rounding instead of two)

## Inference Engine Optimizations

### Buffer Pre-Allocation

`InferenceEngine` chains `output_shape()` calls through all layers at load time to pre-allocate intermediate buffers. This eliminates **all heap allocations** during inference.

```cpp
// At load time: allocate once
for (layer in model) {
    current_shape = layer.output_shape(current_shape);
    buffers.emplace_back(current_shape);
}

// At inference time: zero allocations
layer[0].forward(input, buffers[0]);
layer[1].forward(buffers[0], buffers[1]);
// ...
```

### Warmup Runs

Use warmup runs to stabilize branch prediction and instruction cache before measuring latency:

```cpp
auto model = ModelHandle::Builder()
    .setModelPath("model.titan")
    .enableProfiling()
    .setWarmupRuns(5)    // 5 silent runs before profiling starts
    .build();
```

Warmup runs execute the full forward pass but reset profiling counters afterward.

## Profiling

### InferenceStats

When profiling is enabled, `InferenceStats` tracks:

| Field | Description |
|---|---|
| `inference_count` | Total number of `predict()` calls |
| `total_time_ms` | Cumulative inference time |
| `min_latency_ms` | Fastest single inference |
| `max_latency_ms` | Slowest single inference |
| `mean_latency_ms` | Average latency |
| `layer_times_ms` | Per-layer cumulative time (vector) |

```cpp
auto stats = model.stats();
std::cout << "Mean: " << stats.mean_latency_ms << " ms\n";
std::cout << "Min:  " << stats.min_latency_ms << " ms\n";
std::cout << "Max:  " << stats.max_latency_ms << " ms\n";

// Per-layer breakdown
for (size_t i = 0; i < stats.layer_times_ms.size(); ++i) {
    std::cout << "Layer " << i << ": " << stats.layer_times_ms[i] << " ms\n";
}
```

### Reset and Measure

```cpp
model.reset_stats();

for (int i = 0; i < 1000; ++i) {
    model.predict(input);
}

auto stats = model.stats();
// stats now reflect only the 1000 measured runs
```

## Benchmarking

### Running Benchmarks

```bash
# Compare naive vs AVX2 at 256×256
./tests/matrix_benchmark --benchmark_filter='(Naive_256|AVX2_256)'

# Full suite with statistical repetitions
./tests/matrix_benchmark --benchmark_repetitions=5

# Specific size
./tests/matrix_benchmark --benchmark_filter='1024'
```

### Interpreting Results

Google Benchmark reports:
- **Time**: wall-clock time per iteration
- **CPU**: CPU time per iteration (excludes I/O waits)
- **Iterations**: number of runs for statistical stability

The benchmark automatically selects iteration count to achieve stable measurements.

## Production Deployment Tips

### Model Loading

Load models once and reuse the `ModelHandle`. Model loading involves file I/O, parsing, and buffer pre-allocation — it is not designed to be called per-request.

```cpp
// Good: load once
auto model = ModelHandle::Builder()
    .setModelPath("model.titan")
    .build();

// Use repeatedly
for (auto& request : requests) {
    auto result = model.predict(request.input);
}
```

### Thread Count

`ModelHandle` is thread-safe but serializes access internally via mutex. For maximum throughput with multiple threads, consider:

1. **Single model, multiple threads**: Simple, correct, but serialized. Good when inference is fast relative to other work.
2. **Multiple models**: Load the same model into separate `ModelHandle` instances for true parallel execution. Each has independent buffers.

### Batch Sizing

`predict_batch()` processes inputs sequentially. For large batches, consider splitting across multiple threads with separate model handles:

```cpp
// Better throughput for large batches
auto model1 = load_model("model.titan");
auto model2 = load_model("model.titan");

std::thread t1([&]{ model1.predict_batch(first_half); });
std::thread t2([&]{ model2.predict_batch(second_half); });
t1.join();
t2.join();
```

### Memory Footprint

Per `ModelHandle`:
- Model weights (Dense layer parameters)
- Pre-allocated intermediate buffers (one per layer)
- One `std::mutex`

For a Dense(4,8) → ReLU → Dense(8,3) → Softmax model: ~500 bytes total.
