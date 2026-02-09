#include <benchmark/benchmark.h>
#include "titaninfer/ops/matrix_ops.hpp"
#include "titaninfer/ops/matrix_ops_simd.hpp"

using namespace titaninfer;
using namespace titaninfer::ops;

// ========================================
// Naive Implementation Benchmarks
// ========================================

static void BM_MatMul_Naive_64(benchmark::State& state) {
    Tensor A({64, 64});
    Tensor B({64, 64});
    Tensor C({64, 64});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    // Report GFLOPS
    state.SetItemsProcessed(state.iterations() * 64 * 64 * 64 * 2);
}
BENCHMARK(BM_MatMul_Naive_64);

static void BM_MatMul_Naive_128(benchmark::State& state) {
    Tensor A({128, 128});
    Tensor B({128, 128});
    Tensor C({128, 128});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 128 * 128 * 128 * 2);
}
BENCHMARK(BM_MatMul_Naive_128);

static void BM_MatMul_Naive_256(benchmark::State& state) {
    Tensor A({256, 256});
    Tensor B({256, 256});
    Tensor C({256, 256});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 256 * 256 * 256 * 2);
}
BENCHMARK(BM_MatMul_Naive_256);

static void BM_MatMul_Naive_512(benchmark::State& state) {
    Tensor A({512, 512});
    Tensor B({512, 512});
    Tensor C({512, 512});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 512 * 512 * 512 * 2);
}
BENCHMARK(BM_MatMul_Naive_512);

// ========================================
// AVX2 Implementation Benchmarks
// ========================================

static void BM_MatMul_AVX2_64(benchmark::State& state) {
    Tensor A({64, 64});
    Tensor B({64, 64});
    Tensor C({64, 64});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        simd::matmul_avx2(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 64 * 64 * 64 * 2);
}
BENCHMARK(BM_MatMul_AVX2_64);

static void BM_MatMul_AVX2_128(benchmark::State& state) {
    Tensor A({128, 128});
    Tensor B({128, 128});
    Tensor C({128, 128});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        simd::matmul_avx2(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 128 * 128 * 128 * 2);
}
BENCHMARK(BM_MatMul_AVX2_128);

static void BM_MatMul_AVX2_256(benchmark::State& state) {
    Tensor A({256, 256});
    Tensor B({256, 256});
    Tensor C({256, 256});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        simd::matmul_avx2(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 256 * 256 * 256 * 2);
}
BENCHMARK(BM_MatMul_AVX2_256);

static void BM_MatMul_AVX2_512(benchmark::State& state) {
    Tensor A({512, 512});
    Tensor B({512, 512});
    Tensor C({512, 512});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        simd::matmul_avx2(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 512 * 512 * 512 * 2);
}
BENCHMARK(BM_MatMul_AVX2_512);

// ========================================
// Rectangular Matrix Benchmarks
// ========================================

static void BM_MatMul_Naive_Rect(benchmark::State& state) {
    Tensor A({256, 512});
    Tensor B({512, 128});
    Tensor C({256, 128});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        matmul(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 256 * 512 * 128 * 2);
}
BENCHMARK(BM_MatMul_Naive_Rect);

static void BM_MatMul_AVX2_Rect(benchmark::State& state) {
    Tensor A({256, 512});
    Tensor B({512, 128});
    Tensor C({256, 128});
    A.fill(1.0f);
    B.fill(2.0f);
    
    for (auto _ : state) {
        simd::matmul_avx2(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    
    state.SetItemsProcessed(state.iterations() * 256 * 512 * 128 * 2);
}
BENCHMARK(BM_MatMul_AVX2_Rect);

BENCHMARK_MAIN();
