import os, sys, os.path as osp, time, argparse, random, torch, csv
ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from tokenizers import Tokenizer
from model import TinyLM, build_sincos, prealloc_kvcache

WARMUP = 10

def make_ids(tok, L, device, seed=0, prompt="Once upon a time"):
    random.seed(seed); torch.manual_seed(seed)
    base = tok.encode(prompt).ids
    if len(base) >= L: ids = base[:L]
    else:
        extra = torch.randint(0, tok.get_vocab_size(), (L - len(base),), dtype=torch.long).tolist()
        ids = base + extra
    return torch.tensor(ids, device=device)[None, :]

def measure_with_kv(m, ids, steps, sin, cos, cfg, dtype):
    dhead = cfg['dim']//cfg['n_heads']
    cache = prealloc_kvcache(1, ids.size(1)+WARMUP+steps, cfg['n_heads'], dhead, ids.device.type, dtype)
    _ = m(ids, sin, cos, cache, start_pos=0)  # prefill full prefix
    # warmup incremental
    for _ in range(WARMUP):
        logits = m(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)[:, -1, :]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps):
        logits = m(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)[:, -1, :]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
    torch.cuda.synchronize(); t1 = time.time()
    return steps/(t1-t0)

def measure_no_kv(m, ids, steps, sin, cos, cfg, dtype):
    """Measure throughput without KV-cache by recomputing full sequence each time."""
    # warmup - process full sequence without cache
    tmp = ids.clone()
    for _ in range(3):
        # Process entire sequence without cache (cache=None means no caching)
        logits = m(tmp, sin, cos, cache=None, start_pos=0)[:, -1, :]
        tmp = torch.cat([tmp, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

    torch.cuda.synchronize()
    t0 = time.time()

    # Actual measurement
    tmp = ids.clone()
    for _ in range(steps):
        # Process entire sequence from scratch each time (no cache)
        logits = m(tmp, sin, cos, cache=None, start_pos=0)[:, -1, :]
        tmp = torch.cat([tmp, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

    torch.cuda.synchronize()
    t1 = time.time()
    return steps/(t1-t0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--lengths', type=int, nargs='+', default=[32,64,128,192,256,320,384,448])
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16','fp32'])
    ap.add_argument('--label', type=str, default='gpu')
    ap.add_argument('--prompt', type=str, default='Once upon a time')
    ap.add_argument('--out', type=str, default='out/kv_curve.csv')
    args = ap.parse_args()

    os.makedirs('out', exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    tok = Tokenizer.from_str(ckpt['tok'])
    cfg = ckpt.get('config') or {'dim':384,'n_layers':6,'n_heads':6,'vocab_size':tok.get_vocab_size()}

    m = TinyLM(cfg['vocab_size'], cfg['dim'], cfg['n_layers'], cfg['n_heads']).cuda().eval()
    st = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in st): st = {k.replace('_orig_mod.','',1): v for k,v in st.items()}
    m.load_state_dict(st, strict=False)
    if args.dtype == 'fp16': m.half()

    dtype = next(m.parameters()).dtype
    dhead = cfg['dim']//cfg['n_heads']
    maxL = max(args.lengths) + args.steps + WARMUP + 8
    sin, cos = build_sincos(maxL, dhead, 'cuda'); sin, cos = sin.to(dtype), cos.to(dtype)

    rows = [('label','dtype','context_len','mode','tokens_per_sec')]
    for L in args.lengths:
        try:
            ids = make_ids(tok, L, device='cuda', seed=0, prompt=args.prompt)
            kv   = measure_with_kv(m, ids.clone(), args.steps, sin, cos, cfg, dtype)
            nokv = measure_no_kv(m, ids.clone(), args.steps, sin, cos, cfg, dtype)
            print(f'L={L:4d}  with_kv={kv:.1f} tok/s   no_kv={nokv:.1f} tok/s   speedup={kv/max(nokv,1e-9):.2f}x')
            rows.append((args.label, args.dtype, L, 'with_kv', f'{kv:.3f}'))
            rows.append((args.label, args.dtype, L, 'no_kv',   f'{nokv:.3f}'))
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'L={L}: OOM; skipping')
                torch.cuda.empty_cache()
            else:
                raise

    with open(args.out, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print('Wrote', args.out)
