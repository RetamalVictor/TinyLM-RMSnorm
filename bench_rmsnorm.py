import torch, time
from model import RMSNormCUDA

@torch.no_grad()
def bench(B=16, T=256, C=1024, iters=200, dtype=torch.float16):
    x = torch.randn(B, T, C, device='cuda', dtype=dtype)
    ref_w = torch.ones(C, device='cuda', dtype=dtype)
    mod = RMSNormCUDA(C).cuda().to(dtype)
    mod.weight.copy_(ref_w)

    # Warmup
    for _ in range(50):
        y = mod(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        y = mod(x)
    torch.cuda.synchronize()
    t1 = time.time()
    print({'shape': (B,T,C), 'iters': iters, 'ms_per_iter': (t1-t0)*1000/iters})

if __name__ == '__main__':
    bench()