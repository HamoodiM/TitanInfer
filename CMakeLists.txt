cmake_minimum_required(VERSION 3.14)
project(TitanInfer VERSION 0.1.0 LANGUAGES CXX)

# ============================================================
# Compiler Requirements
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Compiler Flags
# ============================================================
if(MSVC)
    add_compile_options(/W4 /WX /arch:AVX2)
    # Debug mode bounds checking
    add_compile_definitions($<$<CONFIG:Debug>:TITANINFER_DEBUG>)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror
        -mavx2 -mfma
        -march=native
    )
    add_compile_definitions($<$<CONFIG:Debug>:TITANINFER_DEBUG>)
endif()

# ============================================================
# Library Target
# ============================================================
add_library(titaninfer
    src/tensor.cpp
    src/ops/matrix_ops.cpp
)

target_include_directories(titaninfer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ============================================================
# Testing
# ============================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # For Windows: Prevent overriding parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Test executable
    add_executable(tensor_test
        tests/tensor_test.cpp
    )
    add_executable(matrix_ops_test tests/ops/matrix_ops_test.cpp)  
    
    target_link_libraries(tensor_test
        PRIVATE
        titaninfer
        GTest::gtest_main
    )

    target_link_libraries(matrix_ops_test  # <-- ADD THIS
        PRIVATE titaninfer GTest::gtest_main
    )
    
    include(GoogleTest)
    gtest_discover_tests(tensor_test)
    gtest_discover_tests(matrix_ops_test)
endif()

# ============================================================
# Installation
# ============================================================
install(TARGETS titaninfer
    EXPORT TitanInferTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/titaninfer
    DESTINATION include
)
