# ============================================================
# Always-built utilities
# ============================================================

# gen_test_model: writes a minimal .titan file for integration / chaos tests.
add_executable(gen_test_model gen_test_model.cpp)
target_link_libraries(gen_test_model PRIVATE titaninfer)

# ============================================================
# Cluster server binaries — compiled only with USE_GRPC=ON
# Requires: libgrpc++-dev protobuf-compiler-grpc (apt-installed in Docker)
# ============================================================

if(NOT USE_GRPC)
    message(STATUS "USE_GRPC=OFF: skipping cluster server binaries.")
    return()
endif()

# ---- Find gRPC and Protobuf ------------------------------------------------
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND)
    # Fallback: find via pkg-config (common on Ubuntu with apt-installed gRPC).
    # Synthesize gRPC::grpc++ and gRPC::grpc++_reflection IMPORTED targets so
    # that all downstream target_link_libraries() calls work unchanged.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPCPP REQUIRED grpc++>=1.30)

    add_library(gRPC::grpc++ INTERFACE IMPORTED)
    set_target_properties(gRPC::grpc++ PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GRPCPP_INCLUDE_DIRS}"
        INTERFACE_LINK_DIRECTORIES   "${GRPCPP_LIBRARY_DIRS}"
        INTERFACE_LINK_LIBRARIES     "${GRPCPP_LIBRARIES}"
        INTERFACE_COMPILE_OPTIONS    "${GRPCPP_CFLAGS_OTHER}"
    )

    pkg_check_modules(GRPCPP_REFL grpc++_reflection)
    add_library(gRPC::grpc++_reflection INTERFACE IMPORTED)
    if(GRPCPP_REFL_FOUND)
        set_target_properties(gRPC::grpc++_reflection PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GRPCPP_REFL_INCLUDE_DIRS}"
            INTERFACE_LINK_DIRECTORIES   "${GRPCPP_REFL_LIBRARY_DIRS}"
            INTERFACE_LINK_LIBRARIES     "${GRPCPP_REFL_LIBRARIES}"
        )
    else()
        # Library name only — linker will resolve it from GRPCPP dirs above.
        set_target_properties(gRPC::grpc++_reflection PROPERTIES
            INTERFACE_LINK_LIBRARIES "grpc++_reflection"
        )
    endif()
endif()

# ---- Proto codegen ---------------------------------------------------------
set(PROTO_SRC "${CMAKE_SOURCE_DIR}/proto/controller.proto")
set(PROTO_OUT "${CMAKE_CURRENT_BINARY_DIR}")

find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

set(PROTO_CC  "${PROTO_OUT}/controller.pb.cc")
set(PROTO_H   "${PROTO_OUT}/controller.pb.h")
set(GRPC_CC   "${PROTO_OUT}/controller.grpc.pb.cc")
set(GRPC_H    "${PROTO_OUT}/controller.grpc.pb.h")

add_custom_command(
    OUTPUT  ${PROTO_CC} ${PROTO_H} ${GRPC_CC} ${GRPC_H}
    COMMAND ${PROTOC}
            --proto_path="${CMAKE_SOURCE_DIR}/proto"
            --proto_path="${Protobuf_INCLUDE_DIRS}"
            --cpp_out="${PROTO_OUT}"
            --grpc_out="${PROTO_OUT}"
            "--plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}"
            "${PROTO_SRC}"
    DEPENDS "${PROTO_SRC}"
    COMMENT "Generating protobuf/gRPC stubs for controller.proto"
)

add_library(controller_proto STATIC ${PROTO_CC} ${GRPC_CC})
target_include_directories(controller_proto PUBLIC "${PROTO_OUT}")
target_link_libraries(controller_proto PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
)

# ---- cluster_controller_server ---------------------------------------------
add_executable(cluster_controller_server cluster_controller_main.cpp)
target_compile_definitions(cluster_controller_server PRIVATE TITANINFER_USE_GRPC)
target_include_directories(cluster_controller_server
    PRIVATE "${PROTO_OUT}" "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(cluster_controller_server
    PRIVATE titaninfer controller_proto gRPC::grpc++ gRPC::grpc++_reflection)

# ---- model_server_node -----------------------------------------------------
add_executable(model_server_node model_server_node_main.cpp)
target_compile_definitions(model_server_node PRIVATE TITANINFER_USE_GRPC)
target_include_directories(model_server_node
    PRIVATE "${PROTO_OUT}" "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(model_server_node
    PRIVATE titaninfer controller_proto gRPC::grpc++)
