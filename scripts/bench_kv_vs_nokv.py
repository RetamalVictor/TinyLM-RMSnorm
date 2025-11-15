import time, argparse, os, sys, os.path as osp, torch
ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from tokenizers import Tokenizer
from model import TinyLM, build_sincos, prealloc_kvcache

WARMUP = 20

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--steps', type=int, default=256)
ap.add_argument('--prompt', type=str, default='Once upon a time')
ap.add_argument('--label', type=str, default='gpu')
ap.add_argument('--out', type=str, default='out/kv_vs_nokv.csv')
ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16','fp32'])
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location='cpu')
tok = Tokenizer.from_str(ckpt['tok'])
cfg = ckpt.get('config') or {'dim':384,'n_layers':6,'n_heads':6,'vocab_size':tok.get_vocab_size()}

m = TinyLM(cfg['vocab_size'], cfg['dim'], cfg['n_layers'], cfg['n_heads']).cuda().eval()
st = ckpt['model']
if any(k.startswith('_orig_mod.') for k in st): st = {k.replace('_orig_mod.','',1):v for k,v in st.items()}
m.load_state_dict(st, strict=False)
if args.dtype == 'fp16': m.half()

dhead = cfg['dim'] // cfg['n_heads']
dtype = next(m.parameters()).dtype
# build RoPE in model dtype
sin, cos = build_sincos(8192, dhead, 'cuda'); sin, cos = sin.to(dtype), cos.to(dtype)
ids0 = torch.tensor(tok.encode(args.prompt).ids, device='cuda')[None,:]

def with_kv():
    ids = ids0.clone()
    # 1) prefill: write the whole prefix to cache in one pass
    cache = prealloc_kvcache(1, ids.size(1)+WARMUP+args.steps, cfg['n_heads'], dhead, 'cuda', dtype)
    _ = m(ids, sin, cos, cache, start_pos=0)  # prefill prefix
    # 2) warm up incremental
    for _ in range(WARMUP):
        logits = m(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)[:, -1, :]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    # 3) timed incremental
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(args.steps):
        logits = m(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)[:, -1, :]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    torch.cuda.synchronize(); t1=time.time()
    return args.steps/(t1-t0)

def no_kv():
    ids = ids0.clone()
    # recompute over the full prefix each step (no cache reuse)
    # warmup
    for _ in range(5):
        # Process entire sequence without cache (cache=None means no caching)
        logits = m(ids, sin, cos, cache=None, start_pos=0)[:, -1, :]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

    torch.cuda.synchronize()
    t0 = time.time()

    # Actual measurement - process full sequence from scratch each time
    for _ in range(args.steps):
        logits = m(ids, sin, cos, cache=None, start_pos=0)[:, -1, :]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

    torch.cuda.synchronize()
    t1 = time.time()
    return args.steps/(t1-t0)

os.makedirs('out', exist_ok=True)
kv_tps, nokv_tps = with_kv(), no_kv()

hdr = 'label,mode,steps,dtype,tokens_per_sec\n'
append = os.path.exists(args.out)
with open(args.out, 'a' if append else 'w') as f:
    if not append: f.write(hdr)
    f.write(f'{args.label},with_kv,{args.steps},{args.dtype},{kv_tps:.2f}\n')
    f.write(f'{args.label},no_kv,{args.steps},{args.dtype},{nokv_tps:.2f}\n')
print('with_kv tokens/sec:', kv_tps)
print('no_kv  tokens/sec:', nokv_tps)
print('Wrote', args.out)
