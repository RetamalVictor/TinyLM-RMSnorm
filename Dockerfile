FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /workspace
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

ENV HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    UV_SYSTEM_PYTHON=1

# Copy bittorch first (dependency)
COPY bittorch /workspace/bittorch

# Copy TinyLM project
COPY TinyLM-RMSnorm /workspace/app
WORKDIR /workspace/app

# Install with uv, using system torch (already in base image)
# Build CUDA extension explicitly
RUN uv pip install --system -e /workspace/bittorch && \
    uv pip install --system -e ".[dev]" && \
    python setup.py build_ext --inplace

CMD ["bash"]
