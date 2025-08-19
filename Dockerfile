FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /workspace/app
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt requests
COPY . .

ENV HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# RUN python setup_cuda.py build_ext --inplace
CMD ["bash"]