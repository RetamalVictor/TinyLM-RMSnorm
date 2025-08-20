import argparse, csv, matplotlib.pyplot as plt, os, os.path as osp
ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='out/vram_seq.csv')
ap.add_argument('--out', default='out/fig_vram_seq.png')
args = ap.parse_args()

L, MB = [], []
with open(args.csv) as f:
    for r in csv.DictReader(f):
        L.append(int(r['seq_len'])); MB.append(float(r['mb']))

plt.figure()
plt.plot(L, MB, marker='o')
plt.xlabel('sequence length (T_max)'); plt.ylabel('VRAM (MB)')
plt.title('KV-cache VRAM vs sequence length')
plt.tight_layout()

os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=160)
plt.savefig(base + ".svg")
print('Wrote', args.out, 'and', base + '.svg')
