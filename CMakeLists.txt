cmake_minimum_required(VERSION 3.14)
project(TitanInfer VERSION 0.1.0 LANGUAGES C CXX)

# ============================================================
# Compiler Requirements
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Compiler Flags & SIMD Detection
# ============================================================
option(ENABLE_SIMD "Enable SIMD compiler flags (AVX2/FMA). Turn off for non-x86 targets." ON)

set(SIMD_AVAILABLE OFF)

if(MSVC)
    add_compile_options(/W4 /WX)
    if(ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(TITANINFER_ENABLE_SIMD)
        set(SIMD_AVAILABLE ON)
    endif()
    add_compile_definitions($<$<CONFIG:Debug>:TITANINFER_DEBUG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    if(ENABLE_SIMD)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_MAVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_MAVX2 AND COMPILER_SUPPORTS_FMA)
            add_compile_options(-mavx2 -mfma -march=native)
            add_compile_definitions(TITANINFER_ENABLE_SIMD)
            set(SIMD_AVAILABLE ON)
        else()
            message(STATUS "SIMD flags (-mavx2 -mfma) not supported; building without them.")
        endif()
    endif()
    add_compile_definitions($<$<CONFIG:Debug>:TITANINFER_DEBUG>)
endif()

# ============================================================
# Subdirectories
# ============================================================
add_subdirectory(src)

option(BUILD_TESTS "Build unit tests and benchmarks" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(BUILD_EXAMPLES "Build example executables" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
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
