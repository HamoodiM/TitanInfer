# ============================================================
# TitanInfer Docker Build Environment - Phase 4
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Create working directory
WORKDIR /titaninfer

# Copy source code
COPY . .

# Build the project
RUN mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc)

# Entrypoint: CPU detection + tests + benchmarks
CMD ["sh", "-c", "\
    echo '========================================' && \
    echo 'CPU Feature Detection' && \
    echo '========================================' && \
    grep -E 'model name|flags' /proc/cpuinfo | head -2 && \
    echo '' && \
    grep -o 'avx2' /proc/cpuinfo | head -1 && \
    grep -o 'fma' /proc/cpuinfo | head -1 && \
    echo '' && \
    echo '========================================' && \
    echo 'Running Unit Tests' && \
    echo '========================================' && \
    cd build && ctest --output-on-failure && \
    echo '' && \
    echo '========================================' && \
    echo 'Running Performance Benchmarks' && \
    echo '========================================' && \
    ./matrix_benchmark --benchmark_filter='(Naive_256|AVX2_256)' --benchmark_repetitions=3 \
"]
