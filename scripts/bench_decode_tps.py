import time, argparse, os, torch
from tokenizers import Tokenizer
from model import TinyLM, build_sincos, prealloc_kvcache

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--steps', type=int, default=256)
ap.add_argument('--prompt', type=str, default='Once upon a time')
ap.add_argument('--label', type=str, default='gpu')
ap.add_argument('--out', type=str, default='out/decode_bench.csv')
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location='cpu')
tok = Tokenizer.from_str(ckpt['tok'])
cfg = ckpt.get('config') or {'dim':384,'n_layers':6,'n_heads':6,'vocab_size':tok.get_vocab_size()}

m = TinyLM(cfg['vocab_size'], cfg['dim'], cfg['n_layers'], cfg['n_heads']).cuda().eval()
st = ckpt['model']
if any(k.startswith('_orig_mod.') for k in st): st = {k.replace('_orig_mod.','',1):v for k,v in st.items()}
m.load_state_dict(st, strict=False)

dtype = next(m.parameters()).dtype
sin,cos = build_sincos(8192, m.dim//m.n_heads, 'cuda')
sin,cos = sin.to(dtype), cos.to(dtype)
ids = torch.tensor(tok.encode(args.prompt).ids, device='cuda')[None,:]
cache = prealloc_kvcache(1, ids.size(1)+args.steps, m.n_heads, m.dim//m.n_heads, 'cuda', next(m.parameters()).dtype)

for _ in range(20):
    logits = m(ids[:,-1:], sin, cos, cache, start_pos=ids.size(1)-1)[:,-1,:]
    ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)

torch.cuda.synchronize(); t0=time.time()
with torch.no_grad():
    for _ in range(args.steps):
        logits = m(ids[:,-1:], sin, cos, cache, start_pos=ids.size(1)-1)[:,-1,:]
        ids = torch.cat([ids, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
torch.cuda.synchronize(); t1=time.time()
tps = args.steps/(t1-t0)

os.makedirs('out', exist_ok=True)
hdr = 'label,steps,tokens_per_sec\n'
append = os.path.exists(args.out)
with open(args.out, 'a' if append else 'w') as f:
    if not append: f.write(hdr)
    f.write(f'{args.label},{args.steps},{tps:.2f}\n')
print('tokens/sec:', tps, '→ appended to', args.out)
