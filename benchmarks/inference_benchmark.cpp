#include <benchmark/benchmark.h>
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/activation_layer.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/layers/fused_layers.hpp"
#include "titaninfer/layers/quantized_dense_layer.hpp"
#include "titaninfer/engine/model_compiler.hpp"

using namespace titaninfer;
using namespace titaninfer::layers;
using namespace titaninfer::engine;

namespace {

std::unique_ptr<Sequential> make_mlp(size_t input_size, size_t hidden_size,
                                      size_t output_size) {
    auto model = std::make_unique<Sequential>();

    auto d1 = std::make_unique<DenseLayer>(input_size, hidden_size, true);
    Tensor w1({hidden_size, input_size});
    for (size_t i = 0; i < w1.size(); ++i) w1.data()[i] = 0.01f;
    d1->set_weights(w1);
    Tensor b1({hidden_size}); b1.fill(0.01f);
    d1->set_bias(b1);
    model->add(std::move(d1));
    model->add(std::make_unique<ReluLayer>());

    auto d2 = std::make_unique<DenseLayer>(hidden_size, output_size, true);
    Tensor w2({output_size, hidden_size});
    for (size_t i = 0; i < w2.size(); ++i) w2.data()[i] = 0.01f;
    d2->set_weights(w2);
    Tensor b2({output_size}); b2.fill(0.01f);
    d2->set_bias(b2);
    model->add(std::move(d2));

    return model;
}

} // anonymous namespace

static void BM_MLP_FP32_Single(benchmark::State& state) {
    auto model = make_mlp(256, 128, 10);
    Tensor input({256});
    input.fill(1.0f);

    for (auto _ : state) {
        Tensor output = model->forward(input);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MLP_FP32_Single);

static void BM_MLP_FP32_Batch(benchmark::State& state) {
    const size_t batch = static_cast<size_t>(state.range(0));
    auto model = make_mlp(256, 128, 10);
    Tensor input({batch, 256});
    input.fill(1.0f);

    for (auto _ : state) {
        Tensor output = model->forward(input);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch));
}
BENCHMARK(BM_MLP_FP32_Batch)->Arg(1)->Arg(4)->Arg(16)->Arg(64);

static void BM_MLP_Fused(benchmark::State& state) {
    auto model = make_mlp(256, 128, 10);
    CompileOptions opts;
    opts.enable_fusion = true;
    opts.enable_quantization = false;
    auto compiled = ModelCompiler::compile(*model, {256}, opts);

    Tensor input({256});
    input.fill(1.0f);

    for (auto _ : state) {
        Tensor output = compiled.predict(input);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MLP_Fused);

static void BM_MLP_Quantized(benchmark::State& state) {
    auto model = make_mlp(256, 128, 10);
    CompileOptions opts;
    opts.enable_fusion = false;
    opts.enable_quantization = true;
    auto compiled = ModelCompiler::compile(*model, {256}, opts);

    Tensor input({256});
    input.fill(1.0f);

    for (auto _ : state) {
        Tensor output = compiled.predict(input);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MLP_Quantized);

static void BM_CompiledModel(benchmark::State& state) {
    auto model = make_mlp(256, 128, 10);
    auto compiled = ModelCompiler::compile(*model, {256});

    Tensor input({256});
    input.fill(1.0f);

    for (auto _ : state) {
        Tensor output = compiled.predict(input);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CompiledModel);

BENCHMARK_MAIN();
