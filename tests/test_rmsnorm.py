import torch
from model import RMSNormCUDA

def test_forward_close():
    torch.manual_seed(0)
    B,T,C = 4,8,64
    x = torch.randn(B,T,C, device='cuda', dtype=torch.float32, requires_grad=True)
    w = torch.randn(C, device='cuda', dtype=torch.float32)

    # Reference
    with torch.no_grad():
        rms = (x.detach()**2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    y_ref = x * rms * w

    mod = RMSNormCUDA(C).cuda().float()
    mod.weight.data.copy_(w)
    y = mod(x)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4)

def test_backward_close():
    torch.manual_seed(0)
    B,T,C = 2,4,128
    x = torch.randn(B,T,C, device='cuda', dtype=torch.float32, requires_grad=True)
    w = torch.randn(C, device='cuda', dtype=torch.float32, requires_grad=True)

    # Our module
    mod = RMSNormCUDA(C).cuda().float()
    mod.weight.data.copy_(w.detach())
    y = mod(x)
    loss = y.sum()
    dx, dw = torch.autograd.grad(loss, [x, mod.weight])

    # Reference via PyTorch ops
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    rms = (x2**2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    y2 = x2 * rms * w2
    loss2 = y2.sum()
    dx2, dw2 = torch.autograd.grad(loss2, [x2, w2])

    assert torch.allclose(dx, dx2, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dw, dw2, atol=1e-3, rtol=1e-3)