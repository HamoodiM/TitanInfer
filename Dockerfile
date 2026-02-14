# ============================================================
# TitanInfer Multi-Stage Docker Build
# ============================================================

# --- Builder Stage ---
# Full build environment: compiles, tests, and benchmarks
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

WORKDIR /titaninfer
COPY . .

# Build with Release optimizations
RUN mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc)

# Run tests during build to catch failures early
RUN cd build && ctest --output-on-failure

# Default: CPU detection + tests + benchmarks
CMD ["sh", "-c", "\
    echo '========================================' && \
    echo 'TitanInfer Build Container' && \
    echo '========================================' && \
    echo '' && \
    echo 'CPU Features:' && \
    grep -E 'model name' /proc/cpuinfo | head -1 && \
    grep -o 'avx2' /proc/cpuinfo | head -1 && \
    grep -o 'fma' /proc/cpuinfo | head -1 && \
    echo '' && \
    echo 'Running Unit Tests...' && \
    cd build && ctest --output-on-failure && \
    echo '' && \
    echo 'Running Benchmarks...' && \
    ./tests/matrix_benchmark --benchmark_filter='(Naive_256|AVX2_256)' --benchmark_repetitions=3 \
"]

# --- Runtime Stage ---
# Minimal image with only the compiled library and headers
FROM ubuntu:22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /titaninfer

# Copy built library and public headers only
COPY --from=builder /titaninfer/include/ include/
COPY --from=builder /titaninfer/build/src/libtitaninfer.a lib/libtitaninfer.a

CMD ["echo", "TitanInfer runtime image -- link against lib/libtitaninfer.a with include/"]
