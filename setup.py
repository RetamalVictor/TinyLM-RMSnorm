"""Setup script for TinyLM with CUDA extension.

The CUDA RMSNorm kernel is built automatically during installation.
For CPU-only installation, set environment variable: TINYLM_NO_CUDA=1
"""

import os
from pathlib import Path

from setuptools import setup

# Set default CUDA architectures if not specified
# Common architectures: 7.5 (Turing), 8.0 (Ampere A100), 8.6 (Ampere RTX 30xx), 8.9 (Ada RTX 40xx), 9.0 (Hopper)
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

# Check if we should build CUDA extension
BUILD_CUDA = (
    os.environ.get("TINYLM_NO_CUDA", "0") != "1"
    and Path("csrc").exists()
)

ext_modules = []
cmdclass = {}

if BUILD_CUDA:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        if torch.cuda.is_available():
            ext_modules.append(
                CUDAExtension(
                    name="tinylm._ext.rmsnorm_cuda",
                    sources=[
                        "csrc/rmsnorm_binding.cpp",
                        "csrc/rmsnorm_cuda.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-O3"],
                        "nvcc": ["-O3", "--use_fast_math"],
                    },
                )
            )
            cmdclass["build_ext"] = BuildExtension
    except ImportError:
        pass

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
