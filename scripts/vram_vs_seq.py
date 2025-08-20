import argparse, os, torch
from tokenizers import Tokenizer
from model import TinyLM, build_sincos, prealloc_kvcache

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', type=str, default='out/vram_seq.csv')
ap.add_argument('--seq', type=int, nargs='+', default=[128,256,512,1024,1536,2048])
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location='cpu'); tok = Tokenizer.from_str(ckpt['tok'])
cfg = ckpt.get('config') or {'dim':384,'n_layers':6,'n_heads':6,'vocab_size':tok.get_vocab_size()}

m = TinyLM(cfg['vocab_size'], cfg['dim'], cfg['n_layers'], cfg['n_heads']).cuda().eval()
st = ckpt['model']
if any(k.startswith('_orig_mod.') for k in st): st = {k.replace('_orig_mod.','',1):v for k,v in st.items()}
m.load_state_dict(st, strict=False)

dtype = next(m.parameters()).dtype
sin,cos = build_sincos(8192, m.dim//m.n_heads, 'cuda')
sin,cos = sin.to(dtype), cos.to(dtype)
lines = [('seq_len','mb')]
for L in args.seq:
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, cfg['vocab_size'], (1,L), device='cuda')
    cache = prealloc_kvcache(1, L, m.n_heads, m.dim//m.n_heads, 'cuda', next(m.parameters()).dtype)
    _ = m(ids[:,-1:], sin, cos, cache, start_pos=L-1)
    mb = torch.cuda.max_memory_allocated()/ (1024**2)
    lines.append((L, f'{mb:.1f}'))
    print(L, mb)

os.makedirs('out', exist_ok=True)
import csv
with open(args.out, 'w', newline='') as f: csv.writer(f).writerows(lines)
print('Wrote', args.out)
