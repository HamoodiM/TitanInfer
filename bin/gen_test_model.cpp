// gen_test_model.cpp
//
// Utility: generates a minimal valid .titan model file for integration tests
// (Docker cluster, chaos tests, etc.).  The model is a 2-layer dense network
// with fixed zero weights — suitable only for verifying that the inference
// engine can load and run a model; not for real prediction.
//
// Usage:
//   gen_test_model [output_path]
//   gen_test_model /models/test.titan

#include "titaninfer/io/model_serializer.hpp"
#include "titaninfer/layers/dense_layer.hpp"
#include "titaninfer/layers/sequential.hpp"
#include "titaninfer/tensor.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

int main(int argc, char* argv[]) {
    const std::string path = (argc > 1) ? argv[1] : "test.titan";

    // Build a tiny model: 4 → 4 → 4 (all-zero weights produce all-zero output).
    // The exact layer sizes don't matter for chaos / integration tests.
    constexpr size_t DIM = 4;

    titaninfer::layers::Sequential model;

    for (int l = 0; l < 2; ++l) {
        auto layer = std::make_unique<titaninfer::layers::DenseLayer>(DIM, DIM);

        // Zero-initialise weights and bias so inference produces no NaN/Inf.
        titaninfer::Tensor w({DIM, DIM});
        float* wp = w.data();
        for (size_t i = 0; i < DIM * DIM; ++i)
            wp[i] = (i % (DIM + 1) == 0) ? 1.0f : 0.0f; // identity matrix

        titaninfer::Tensor b({DIM});
        float* bp = b.data();
        for (size_t i = 0; i < DIM; ++i)
            bp[i] = 0.0f;

        layer->set_weights(w);
        layer->set_bias(b);
        model.add(std::move(layer));
    }

    try {
        titaninfer::io::ModelSerializer::save(model, path);
        std::printf("[gen_test_model] wrote %s (%zu layers)\n",
                    path.c_str(), model.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[gen_test_model] ERROR: %s\n", e.what());
        return 1;
    }
}
