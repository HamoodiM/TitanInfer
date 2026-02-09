# ============================================================
# TitanInfer Docker Build Environment
# ============================================================
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
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

# Set GCC 11 as default (supports C++17 and AVX2)
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

# Default command: run all tests
CMD ["sh", "-c", "cd build && ctest --output-on-failure --verbose"]
