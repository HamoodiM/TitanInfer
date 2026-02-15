# ============================================================
# TitanInfer Library Target
# ============================================================
add_library(titaninfer
    tensor.cpp
    ops/matrix_ops.cpp
    ops/activations.cpp
    ops/conv_ops.cpp
    ops/quantized_ops.cpp
    quantized_tensor.cpp
    layers/activation_layer.cpp
    layers/dense_layer.cpp
    layers/sequential.cpp
    layers/conv2d_layer.cpp
    layers/pooling_layers.cpp
    layers/flatten_layer.cpp
    layers/fused_layers.cpp
    layers/quantized_dense_layer.cpp
    io/model_serializer.cpp
    io/model_parser.cpp
    engine/inference_engine.cpp
    engine/thread_pool.cpp
    engine/fusion.cpp
    engine/dynamic_batcher.cpp
    engine/model_compiler.cpp
    logger.cpp
    model_handle.cpp
    titaninfer_c.cpp
)

# Add SIMD source only if available
if(SIMD_AVAILABLE)
    target_sources(titaninfer PRIVATE ops/matrix_ops_simd.cpp)
endif()

target_include_directories(titaninfer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

# Thread support (pthread on Linux, no-op on Windows)
find_package(Threads REQUIRED)
target_link_libraries(titaninfer PUBLIC Threads::Threads)
