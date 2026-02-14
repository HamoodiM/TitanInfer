/**
 * @file c_api_usage.c
 * @brief Pure C example using the TitanInfer C API
 *
 * Demonstrates loading a model, running inference, and error handling
 * using only C-compatible functions from titaninfer_c.h.
 *
 * Usage:
 *   ./c_api_usage model.titan
 */

#include "titaninfer/titaninfer_c.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.titan>\n", argv[0]);
        return 1;
    }

    /* Load model with input shape {4} */
    size_t input_shape[] = {4};
    TitanInferModelHandle model = titaninfer_load(argv[1], input_shape, 1);

    if (!model) {
        fprintf(stderr, "Failed to load model from '%s'\n", argv[1]);
        return 1;
    }

    printf("Model loaded: %zu layers\n", titaninfer_layer_count(model));

    /* Prepare input data */
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[16];  /* allocate more than needed */
    size_t output_len = 0;

    /* Run inference */
    int status = titaninfer_predict(model, input, 4, output, 16, &output_len);

    if (status != TITANINFER_OK) {
        const char* err = titaninfer_last_error(model);
        fprintf(stderr, "Inference failed (code %d): %s\n",
                status, err ? err : "unknown error");
        titaninfer_free(model);
        return 1;
    }

    /* Print results */
    printf("Output (%zu values):", output_len);
    for (size_t i = 0; i < output_len; ++i) {
        printf(" %.6f", output[i]);
    }
    printf("\n");

    /* Check stats */
    printf("Inference count: %d\n", titaninfer_inference_count(model));

    /* Cleanup */
    titaninfer_free(model);
    return 0;
}
