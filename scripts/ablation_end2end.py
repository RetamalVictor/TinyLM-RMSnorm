import time, argparse, os, torch, torch.nn as nn, random
from tokenizers import Tokenizer
from model import TinyLM, build_sincos, prealloc_kvcache, RMSNormCUDA

class RMSNormRef(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

def build_model_with(norm_cls, cfg):
    import model as mdl
    saved = mdl.RMSNormCUDA
    mdl.RMSNormCUDA = norm_cls
    try:
        m = mdl.TinyLM(cfg['vocab_size'], cfg['dim'], cfg['n_layers'], cfg['n_heads'])
    finally:
        mdl.RMSNormCUDA = saved
    return m

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--steps', type=int, default=256)
ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16','fp32'])
ap.add_argument('--out', type=str, default='out/ablation_rmsnorm.csv')
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
torch.manual_seed(0); random.seed(0)

ckpt = torch.load(args.ckpt, map_location='cpu'); tok = Tokenizer.from_str(ckpt['tok'])
cfg = ckpt.get('config') or {'dim':384,'n_layers':6,'n_heads':6,'vocab_size':tok.get_vocab_size()}

fused = build_model_with(RMSNormCUDA, cfg).cuda().eval()
ref   = build_model_with(RMSNormRef,   cfg).cuda().eval()
st = ckpt['model']
if any(k.startswith('_orig_mod.') for k in st): st = {k.replace('_orig_mod.','',1):v for k,v in st.items()}
fused.load_state_dict(st, strict=False); ref.load_state_dict(st, strict=False)
if args.dtype == 'fp16': fused.half(); ref.half()

dhead = cfg['dim']//cfg['n_heads']
dtype = next(fused.parameters()).dtype
sin,cos = build_sincos(8192, dhead, 'cuda')
sin,cos = sin.to(dtype), cos.to(dtype)
ids0 = torch.tensor(tok.encode('Once upon a time').ids, device='cuda')[None,:]
dtype = next(fused.parameters()).dtype

def ms_per_token(m):
    ids = ids0.clone()
    cache = prealloc_kvcache(1, ids.size(1)+args.steps, cfg['n_heads'], dhead, 'cuda', dtype)
    for _ in range(20):
        logits = m(ids[:,-1:], sin, cos, cache, start_pos=ids.size(1)-1)[:,-1,:]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(args.steps):
        logits = m(ids[:,-1:], sin, cos, cache, start_pos=ids.size(1)-1)[:,-1,:]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    torch.cuda.synchronize(); t1=time.time()
    return (t1-t0)/args.steps*1000.0

ms_ref   = ms_per_token(ref)
ms_fused = ms_per_token(fused)
speedup  = ms_ref / ms_fused if ms_fused>0 else 0.0

print('ref ms/token:', ms_ref)
print('fused ms/token:', ms_fused)
print('speedup x:', speedup)

os.makedirs('out', exist_ok=True)
with open(args.out,'w') as f:
    f.write('variant,ms_per_token\n')
    f.write(f'ref,{ms_ref:.3f}\n')
    f.write(f'fused,{ms_fused:.3f}\n')
print('Wrote', args.out)
