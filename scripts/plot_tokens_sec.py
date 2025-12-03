import argparse
import csv
import os
import os.path as osp

import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='out/decode_bench.csv')
ap.add_argument('--out', default='out/fig_tokens_sec.png')
args = ap.parse_args()

labels, tps = [], []
with open(args.csv) as f:
    for r in csv.DictReader(f):
        labels.append(r['label']); tps.append(float(r['tokens_per_sec']))

plt.figure()
plt.bar(range(len(labels)), tps)
plt.xticks(range(len(labels)), labels, rotation=15)
plt.ylabel('tokens/sec'); plt.title('Decode throughput')
plt.tight_layout()

os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=160)
plt.savefig(base + ".svg")
print('Wrote', args.out, 'and', base + '.svg')
