#!/usr/bin/env bash
# ============================================================
# TitanInfer Test Runner
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
USE_VALGRIND=false
USE_COVERAGE=false

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --valgrind   Run tests under Valgrind for memory leak detection"
    echo "  --coverage   Build with coverage flags and generate lcov report"
    echo "  --help       Show this help message"
}

for arg in "$@"; do
    case "$arg" in
        --valgrind) USE_VALGRIND=true ;;
        --coverage) USE_COVERAGE=true ;;
        --help)     usage; exit 0 ;;
        *)          echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

# Build in Debug mode for bounds checking
CMAKE_EXTRA_FLAGS=""
if [ "$USE_COVERAGE" = true ]; then
    CMAKE_EXTRA_FLAGS="-DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage"
fi

echo "=========================================="
echo "Building in Debug mode..."
echo "=========================================="

mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    $CMAKE_EXTRA_FLAGS
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo ""
echo "=========================================="
echo "Running Unit Tests"
echo "=========================================="

cd "$BUILD_DIR"

if [ "$USE_VALGRIND" = true ]; then
    echo "(with Valgrind memory leak detection)"
    echo ""

    TESTS=(
        tests/tensor_test
        tests/matrix_ops_test
        tests/activations_test
        tests/layer_test
        tests/sequential_test
        tests/serialization_test
        tests/inference_engine_test
        tests/api_test
    )

    # Add SIMD test if it exists
    if [ -f tests/matrix_ops_simd_test ]; then
        TESTS+=(tests/matrix_ops_simd_test)
    fi

    PASS=0
    FAIL=0
    for test in "${TESTS[@]}"; do
        echo "--- $test ---"
        if valgrind --leak-check=full --error-exitcode=1 --quiet "./$test" 2>&1; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            echo "FAILED: $test"
        fi
        echo ""
    done

    echo "=========================================="
    echo "Valgrind Summary: $PASS passed, $FAIL failed"
    echo "=========================================="

    [ "$FAIL" -eq 0 ] || exit 1
else
    ctest --output-on-failure

    if [ "$USE_COVERAGE" = true ]; then
        echo ""
        echo "=========================================="
        echo "Generating Coverage Report"
        echo "=========================================="

        if command -v lcov >/dev/null 2>&1; then
            lcov --capture --directory . --output-file coverage.info --quiet
            lcov --remove coverage.info '/usr/*' '*/googletest/*' '*/googlebenchmark/*' \
                 --output-file coverage_filtered.info --quiet
            if command -v genhtml >/dev/null 2>&1; then
                genhtml coverage_filtered.info --output-directory coverage_report --quiet
                echo "Coverage report: $BUILD_DIR/coverage_report/index.html"
            else
                echo "genhtml not found -- install lcov for HTML reports"
                lcov --list coverage_filtered.info
            fi
        else
            echo "lcov not found -- install lcov for coverage reports"
            echo "  Ubuntu: sudo apt-get install lcov"
            echo "  macOS:  brew install lcov"
        fi
    fi
fi
