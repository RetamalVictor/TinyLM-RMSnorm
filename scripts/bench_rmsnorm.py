import time, csv, argparse
import torch
import torch.nn as nn

# Reference RMSNorm in PyTorch ops
class RMSNormRef(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# Fused one (imported from repo)
from model import RMSNormCUDA

def bench_once(mod, shape, iters=100, dtype=torch.float16, device='cuda'):
    B,T,C = shape
    x = torch.randn(B,T,C, device=device, dtype=dtype)
    # warmup
    for _ in range(10):
        y = mod(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = mod(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='out/rmsnorm_bench.csv')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16','fp32'])
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype=='fp16' else torch.float32
    dev = 'cuda'

    shapes = [(16,256,c) for c in [512,1024,2048]] + [(8,512,1024)]

    rows = [('B','T','C','dtype','op','ms_per_iter')]

    for B,T,C in shapes:
        ref = RMSNormRef(C).to(dev).to(dtype)
        fused = RMSNormCUDA(C).to(dev).to(dtype)
        ms_ref  = bench_once(ref, (B,T,C), args.iters, dtype, dev)
        ms_fused= bench_once(fused, (B,T,C), args.iters, dtype, dev)
        rows.append((B,T,C,args.dtype,'ref',  ms_ref))
        rows.append((B,T,C,args.dtype,'fused',ms_fused))
        print(f"C={C}: ref {ms_ref:.3f} ms, fused {ms_fused:.3f} ms")

    os = __import__('os'); os.makedirs('out', exist_ok=True)
    with open(args.out,'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print('Wrote', args.out)