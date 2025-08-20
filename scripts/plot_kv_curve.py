import argparse, csv, matplotlib.pyplot as plt, os, os.path as osp
ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='out/kv_curve.csv')
ap.add_argument('--out', default='out/fig_kv_curve_panels.png')
ap.add_argument('--label', default=None, help="Filter to a single label (e.g., RTX2070). Default: all")
ap.add_argument('--logx', action='store_true')
ap.add_argument('--dpi', type=int, default=160)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))

# Select labels
all_labels = sorted({r['label'] for r in rows})
labels = [args.label] if args.label else all_labels

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax1, ax2 = axes

for lab in labels:
    rlab = [r for r in rows if r['label'] == lab]
    # Build per-mode dicts: L -> tps
    d_kv   = {}
    d_nokv = {}
    for r in rlab:
        L = int(r['context_len']); tps = float(r['tokens_per_sec'])
        if r['mode'] == 'with_kv': d_kv[L] = tps
        elif r['mode'] == 'no_kv': d_nokv[L] = tps

    # Only lengths present in BOTH series
    Ls = sorted(set(d_kv).intersection(d_nokv))
    if not Ls: 
        continue
    kv   = [d_kv[L]   for L in Ls]
    nokv = [d_nokv[L] for L in Ls]
    speed = [ (a/b) if b>0 else 0.0 for a,b in zip(kv, nokv) ]

    # Panel 1: tokens/sec
    ax1.plot(Ls, kv,   marker='o', label=f'{lab} • KV')
    ax1.plot(Ls, nokv, marker='o', linestyle='--', label=f'{lab} • no-KV')

    # Panel 2: speedup×
    ax2.plot(Ls, speed, marker='o', label=lab)

# Ax cosmetics
for ax in (ax1, ax2):
    if args.logx: ax.set_xscale('log', base=2)
    ax.grid(True, linestyle='--', alpha=0.3)

ax1.set_title('Decode throughput vs context length')
ax1.set_xlabel('context length (tokens)'); ax1.set_ylabel('tokens/sec')
ax1.legend()

ax2.set_title('KV-cache speedup vs context length')
ax2.set_xlabel('context length (tokens)'); ax2.set_ylabel('speedup (×)')
ax2.legend()

plt.tight_layout()
os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=args.dpi)
plt.savefig(base + '.svg')
print('Wrote', args.out, 'and', base + '.svg')
