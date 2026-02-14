#!/usr/bin/env bash
# ============================================================
# TitanInfer Build Script (Linux / macOS)
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
BUILD_TYPE="Release"
RUN_TESTS=false
ENABLE_SIMD="ON"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --debug      Build in Debug mode (enables bounds checking)"
    echo "  --clean      Remove build directory before building"
    echo "  --test       Run tests after building"
    echo "  --no-simd    Disable AVX2/FMA SIMD optimizations"
    echo "  --help       Show this help message"
}

for arg in "$@"; do
    case "$arg" in
        --debug)    BUILD_TYPE="Debug" ;;
        --clean)    rm -rf "$BUILD_DIR" ;;
        --test)     RUN_TESTS=true ;;
        --no-simd)  ENABLE_SIMD="OFF" ;;
        --help)     usage; exit 0 ;;
        *)          echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

mkdir -p "$BUILD_DIR"

# Detect build tool
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="-G Ninja"
    echo "Using Ninja build system"
else
    GENERATOR=""
    echo "Using Make build system"
fi

echo "Build type: $BUILD_TYPE"
echo "SIMD: $ENABLE_SIMD"
echo ""

# Configure
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    $GENERATOR \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_SIMD="$ENABLE_SIMD"

# Build
START_TIME=$(date +%s)
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
END_TIME=$(date +%s)

echo ""
echo "Build completed in $((END_TIME - START_TIME)) seconds"

# Optionally run tests
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "Running tests..."
    cd "$BUILD_DIR" && ctest --output-on-failure
fi
