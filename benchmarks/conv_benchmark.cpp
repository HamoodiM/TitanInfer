#include <benchmark/benchmark.h>
#include "titaninfer/ops/conv_ops.hpp"
#include "titaninfer/layers/conv2d_layer.hpp"

using namespace titaninfer;
using namespace titaninfer::layers;

static void BM_Im2Col(benchmark::State& state) {
    const size_t spatial = static_cast<size_t>(state.range(0));
    const size_t kernel = static_cast<size_t>(state.range(1));
    const size_t channels = 16;

    Tensor input({channels, spatial, spatial});
    input.fill(1.0f);
    Tensor col({1});

    for (auto _ : state) {
        ops::im2col(input, col, kernel, kernel, 1, 1, 0, 0);
        benchmark::DoNotOptimize(col.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Im2Col)
    ->Args({32, 3})->Args({32, 5})->Args({32, 7})
    ->Args({64, 3})->Args({64, 5});

static void BM_Conv2D_Forward(benchmark::State& state) {
    const size_t spatial = static_cast<size_t>(state.range(0));
    const size_t in_ch = 16;
    const size_t out_ch = 32;

    Conv2DLayer conv(in_ch, out_ch, 3, 1, ops::PaddingMode::VALID, false);
    Tensor w({out_ch, in_ch, 3, 3});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.01f;
    conv.set_weights(w);

    Tensor input({in_ch, spatial, spatial});
    input.fill(1.0f);
    Tensor output({1});

    for (auto _ : state) {
        conv.forward(input, output);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Conv2D_Forward)->Arg(28)->Arg(56)->Arg(112);

static void BM_Conv2D_Batched(benchmark::State& state) {
    const size_t batch = static_cast<size_t>(state.range(0));

    Conv2DLayer conv(3, 16, 3, 1, ops::PaddingMode::VALID, true);
    Tensor w({16, 3, 3, 3});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.01f;
    conv.set_weights(w);
    Tensor b({16}); b.fill(0.01f);
    conv.set_bias(b);

    Tensor input({batch, 3, 28, 28});
    input.fill(1.0f);
    Tensor output({1});

    for (auto _ : state) {
        conv.forward(input, output);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch));
}
BENCHMARK(BM_Conv2D_Batched)->Arg(1)->Arg(4)->Arg(16);

BENCHMARK_MAIN();
