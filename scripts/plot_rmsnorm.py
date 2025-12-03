import argparse
import csv
import os
import os.path as osp

import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='out/rmsnorm_bench.csv')
ap.add_argument('--out', default='out/fig_rmsnorm.png')
args = ap.parse_args()

with open(args.csv) as f: rows = list(csv.DictReader(f))
Cs = sorted({int(r['C']) for r in rows})
ref   = [next(float(rr['ms_per_iter']) for rr in rows if int(rr['C'])==c and rr['op']=='ref') for c in Cs]
fused = [next(float(rr['ms_per_iter']) for rr in rows if int(rr['C'])==c and rr['op']=='fused') for c in Cs]

plt.figure()
plt.plot(Cs, ref, marker='o', label='PyTorch RMSNorm')
plt.plot(Cs, fused, marker='o', label='Fused RMSNorm')
plt.xlabel('hidden size C'); plt.ylabel('ms/iter'); plt.title('RMSNorm micro-bench')
plt.legend(); plt.tight_layout()

os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=160)
plt.savefig(base + ".svg")
print('Wrote', args.out, 'and', base + '.svg')
