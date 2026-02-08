# TitanInfer: Production Build Environment
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    gcc-11 \
    g++-11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && rm -rf /var/lib/apt/lists/*

# Verify AVX2 support (will warn if not available)
RUN echo "Checking CPU features..." && \
    grep -o 'avx2' /proc/cpuinfo || echo "Warning: AVX2 not detected (acceptable for cross-compilation)"

# Set working directory
WORKDIR /titaninfer

# Copy source code
COPY . .

# Build in Release mode by default
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . -j$(nproc)

# Run tests to verify build
RUN cd build && ctest --output-on-failure

# Default command
CMD ["/bin/bash"]
