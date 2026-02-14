# ============================================================
# TitanInfer Library Target
# ============================================================
add_library(titaninfer
    tensor.cpp
    ops/matrix_ops.cpp
    ops/activations.cpp
    layers/activation_layer.cpp
    layers/dense_layer.cpp
    layers/sequential.cpp
    io/model_serializer.cpp
    io/model_parser.cpp
    engine/inference_engine.cpp
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
