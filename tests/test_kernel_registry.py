"""Tests for the kernel backend registry system."""

import pytest
import torch

from tinylm.kernels import (
    BACKEND_REGISTRY,
    set_backend,
    get_backend,
    get_kernel,
    available_backends,
    KernelBackend,
)
from tinylm.kernels.backends import PyTorchBackend, CUDABackend, TritonBackend
from tinylm.components.normalization import RMSNorm


class TestKernelBackendRegistry:
    """Tests for the KernelBackendRegistry class."""

    def test_registry_has_default_backends(self):
        """Test that default backends are registered."""
        assert "pytorch" in BACKEND_REGISTRY
        assert "cuda" in BACKEND_REGISTRY
        assert "triton" in BACKEND_REGISTRY

    def test_available_backends(self):
        """Test available_backends returns at least pytorch."""
        backends = available_backends()
        assert "pytorch" in backends
        assert isinstance(backends, list)

    def test_pytorch_always_available(self):
        """PyTorch backend should always be available."""
        assert PyTorchBackend.is_available()

    def test_set_backend_auto(self):
        """Test setting backend to auto."""
        set_backend("auto")
        assert BACKEND_REGISTRY.get_current_backend_name() == "auto"

    def test_set_backend_pytorch(self):
        """Test setting backend to pytorch."""
        set_backend("pytorch")
        assert BACKEND_REGISTRY.get_current_backend_name() == "pytorch"
        backend = get_backend()
        assert backend.name == "pytorch"
        # Reset
        set_backend("auto")

    def test_set_invalid_backend_raises(self):
        """Test that setting an invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("invalid_backend")

    def test_get_backend_for_cpu_device(self):
        """Test that CPU device gets PyTorch backend."""
        set_backend("auto")
        device = torch.device("cpu")
        backend = get_backend(device)
        # CUDA backend doesn't support CPU
        assert backend.name == "pytorch"

    def test_get_kernel_rmsnorm(self):
        """Test getting rmsnorm kernel."""
        set_backend("pytorch")
        kernel = get_kernel("rmsnorm")
        assert kernel is not None
        assert hasattr(kernel, "forward")
        set_backend("auto")

    def test_get_invalid_kernel_raises(self):
        """Test that getting an invalid kernel raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            get_kernel("invalid_kernel")


class TestPyTorchBackend:
    """Tests for the PyTorch backend."""

    def test_is_available(self):
        """PyTorch backend is always available."""
        assert PyTorchBackend.is_available()

    def test_supports_cpu(self):
        """PyTorch backend supports CPU."""
        assert PyTorchBackend.supports_device(torch.device("cpu"))

    def test_supports_cuda(self):
        """PyTorch backend supports CUDA (if available)."""
        if torch.cuda.is_available():
            assert PyTorchBackend.supports_device(torch.device("cuda"))

    def test_rmsnorm_forward(self):
        """Test PyTorch RMSNorm forward pass."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8)
        weight = torch.ones(8)
        eps = 1e-6

        kernel = PyTorchBackend.rmsnorm
        y, inv_rms = kernel.forward(x, weight, eps)

        # Check shape
        assert y.shape == x.shape
        # inv_rms should be None for PyTorch backend
        assert inv_rms is None

        # Verify computation manually
        expected_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected_y = x * expected_rms * weight
        assert torch.allclose(y, expected_y, atol=1e-6)


class TestCUDABackend:
    """Tests for the CUDA backend."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_supports_cuda_device(self):
        """CUDA backend supports CUDA devices (if kernels compiled)."""
        if CUDABackend.is_available():
            assert CUDABackend.supports_device(torch.device("cuda"))

    def test_does_not_support_cpu(self):
        """CUDA backend does not support CPU."""
        assert not CUDABackend.supports_device(torch.device("cpu"))

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not CUDABackend.is_available(),
        reason="CUDA kernel not available"
    )
    def test_rmsnorm_forward_cuda(self):
        """Test CUDA RMSNorm forward pass."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8, device="cuda")
        weight = torch.ones(8, device="cuda")
        eps = 1e-6

        kernel = CUDABackend.rmsnorm
        y, inv_rms = kernel.forward(x, weight, eps)

        # Check shape
        assert y.shape == x.shape
        # CUDA backend returns inv_rms
        assert inv_rms is not None

        # Verify against PyTorch reference
        expected_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected_y = x * expected_rms * weight
        assert torch.allclose(y, expected_y, atol=1e-4)


class TestTritonBackend:
    """Tests for the Triton backend stub."""

    def test_is_not_available_yet(self):
        """Triton backend returns False until implemented."""
        # Triton is a stub, so it should return False
        assert not TritonBackend.is_available()

    def test_rmsnorm_raises_not_implemented(self):
        """Triton RMSNorm raises NotImplementedError."""
        x = torch.randn(2, 4, 8)
        weight = torch.ones(8)

        with pytest.raises(NotImplementedError):
            TritonBackend.rmsnorm.forward(x, weight, 1e-6)


class TestRMSNormWithRegistry:
    """Tests for RMSNorm using the kernel registry."""

    def test_rmsnorm_cpu_uses_pytorch(self):
        """RMSNorm on CPU should use PyTorch backend."""
        set_backend("auto")
        norm = RMSNorm(dim=8)
        x = torch.randn(2, 4, 8)
        y = norm(x)
        assert y.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rmsnorm_cuda_with_auto_backend(self):
        """RMSNorm on CUDA with auto backend."""
        set_backend("auto")
        norm = RMSNorm(dim=8).cuda()
        x = torch.randn(2, 4, 8, device="cuda")
        y = norm(x)
        assert y.shape == x.shape
        assert y.device.type == "cuda"

    def test_rmsnorm_forced_pytorch_backend(self):
        """RMSNorm with forced PyTorch backend."""
        set_backend("pytorch")
        norm = RMSNorm(dim=8)
        x = torch.randn(2, 4, 8)
        y = norm(x)
        assert y.shape == x.shape

        # Verify with reference
        expected_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
        expected_y = x * expected_rms * norm.weight
        assert torch.allclose(y, expected_y, atol=1e-5)

        set_backend("auto")

    def test_rmsnorm_backward(self):
        """Test RMSNorm backward pass."""
        set_backend("pytorch")
        norm = RMSNorm(dim=8)
        x = torch.randn(2, 4, 8, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert norm.weight.grad is not None
        assert norm.weight.grad.shape == norm.weight.shape

        set_backend("auto")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not CUDABackend.is_available(),
        reason="CUDA kernel not available"
    )
    def test_rmsnorm_cuda_backward(self):
        """Test RMSNorm backward pass with CUDA kernel."""
        set_backend("cuda")
        norm = RMSNorm(dim=8).cuda()
        x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert norm.weight.grad is not None

        set_backend("auto")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not CUDABackend.is_available(),
        reason="CUDA kernel not available"
    )
    def test_rmsnorm_cuda_vs_pytorch_equivalence(self):
        """Test that CUDA and PyTorch backends produce same results."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, 64, device="cuda", requires_grad=True)

        # PyTorch backend
        set_backend("pytorch")
        norm_pt = RMSNorm(dim=64).cuda()
        y_pt = norm_pt(x)
        loss_pt = y_pt.sum()
        loss_pt.backward()
        dx_pt = x.grad.clone()
        dw_pt = norm_pt.weight.grad.clone()

        # Reset gradients
        x.grad = None
        norm_pt.weight.grad = None

        # CUDA backend
        set_backend("cuda")
        norm_cuda = RMSNorm(dim=64).cuda()
        norm_cuda.weight.data.copy_(norm_pt.weight.data)
        y_cuda = norm_cuda(x)
        loss_cuda = y_cuda.sum()
        loss_cuda.backward()
        dx_cuda = x.grad
        dw_cuda = norm_cuda.weight.grad

        # Compare
        assert torch.allclose(y_pt, y_cuda, atol=1e-4)
        assert torch.allclose(dx_pt, dx_cuda, atol=1e-3)
        assert torch.allclose(dw_pt, dw_cuda, atol=1e-3)

        set_backend("auto")


class TestFallbackChain:
    """Tests for the fallback chain behavior."""

    def test_fallback_to_pytorch_on_cpu(self):
        """On CPU, should fallback to PyTorch even if CUDA is set."""
        set_backend("auto")
        device = torch.device("cpu")
        backend = get_backend(device)
        assert backend.name == "pytorch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fallback_chain_cuda_to_pytorch(self):
        """If CUDA kernel not compiled, should fallback to PyTorch on GPU."""
        set_backend("auto")
        backends = available_backends()
        # At minimum, pytorch should be available
        assert "pytorch" in backends
