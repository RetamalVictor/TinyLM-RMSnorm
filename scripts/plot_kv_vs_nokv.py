import argparse, csv, matplotlib.pyplot as plt, os, os.path as osp
ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='out/kv_vs_nokv.csv')
ap.add_argument('--out', default='out/fig_kv_vs_nokv.png')
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
labels = sorted({r['label'] for r in rows})
modes  = ['no_kv','with_kv']

plt.figure()
for i, lab in enumerate(labels):
    vals = []
    for mode in modes:
        match = [float(r['tokens_per_sec']) for r in rows if r['label']==lab and r['mode']==mode]
        vals.append(match[0] if match else 0.0)
    x = [i*3 + j for j in range(len(modes))]
    plt.bar(x, vals)
plt.xticks([i*3+0.5 for i in range(len(labels))], labels, rotation=15)
plt.ylabel('tokens/sec'); plt.title('Decode: with-KV vs no-KV')
plt.tight_layout()
os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=160); plt.savefig(base+'.svg'); print('Wrote', args.out, 'and', base+'.svg')
