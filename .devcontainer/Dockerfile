FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /workspace
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git curl ca-certificates vim && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

ENV HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    UV_SYSTEM_PYTHON=1

WORKDIR /workspace/app

CMD ["bash"]
