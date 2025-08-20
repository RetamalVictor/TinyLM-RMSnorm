import argparse, csv, matplotlib.pyplot as plt, os, os.path as osp
ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='out/ablation_rmsnorm.csv')
ap.add_argument('--out', default='out/fig_ablation.png')
args = ap.parse_args()

variants, ms = [], []
with open(args.csv) as f:
    for r in csv.DictReader(f):
        variants.append(r['variant']); ms.append(float(r['ms_per_token']))

plt.figure()
plt.bar(range(len(variants)), ms)
plt.xticks(range(len(variants)), variants)
plt.ylabel('ms/token'); plt.title('End-to-end decode: RMSNorm ref vs fused')
plt.tight_layout()

os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=160)
plt.savefig(base + ".svg")
print('Wrote', args.out, 'and', base + '.svg')
