"""CUDA extensions for TinyLM."""

try:
    from tinylm._ext import rmsnorm_cuda
    HAS_RMSNORM_CUDA = True
except ImportError:
    rmsnorm_cuda = None
    HAS_RMSNORM_CUDA = False
