/**
 * @file simple_inference.cpp
 * @brief Minimal example: load a .titan model and run inference
 *
 * Usage:
 *   ./simple_inference model.titan
 */

#include "titaninfer/TitanInfer.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.titan>\n";
        return 1;
    }

    try {
        // Configure logging
        titaninfer::Logger::instance().set_level(titaninfer::LogLevel::INFO);

        // Load model
        auto model = titaninfer::ModelHandle::Builder()
            .setModelPath(argv[1])
            .enableProfiling()
            .build();

        std::cout << "Model loaded successfully!\n";
        std::cout << model.summary() << "\n";

        // Create input (zeros — replace with real data)
        auto shape = model.expected_input_shape();
        titaninfer::Tensor input(shape);
        input.fill(1.0f);

        // Run inference
        titaninfer::Tensor output = model.predict(input);

        // Print output
        std::cout << "Output (" << output.size() << " values):";
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << " " << output.data()[i];
        }
        std::cout << "\n";

        // Print profiling stats
        auto stats = model.stats();
        std::cout << "Inference latency: " << stats.mean_latency_ms << " ms\n";

    } catch (const titaninfer::ModelLoadException& e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    } catch (const titaninfer::InferenceException& e) {
        std::cerr << "Inference error: " << e.what() << "\n";
        return 1;
    } catch (const titaninfer::TitanInferException& e) {
        std::cerr << "TitanInfer error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
