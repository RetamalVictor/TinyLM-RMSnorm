import argparse, csv, matplotlib.pyplot as plt, os, os.path as osp
ap = argparse.ArgumentParser()
ap.add_argument('--log', default='out/train_log.csv')
ap.add_argument('--out', default='out/fig_training_curve.png')
args = ap.parse_args()

steps, train, vsteps, vvals = [], [], [], []
with open(args.log) as f:
    r = csv.DictReader(f)
    for row in r:
        s = int(row['step']); steps.append(s); train.append(float(row['train_loss']))
        if row['val_loss']:
            vsteps.append(s); vvals.append(float(row['val_loss']))

plt.figure()
plt.plot(steps, train, label='train loss')
if vsteps: plt.plot(vsteps, vvals, label='val loss')
plt.xlabel('step'); plt.ylabel('loss'); plt.title('Training/Validation Loss')
plt.legend(); plt.tight_layout()

os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
base, _ = osp.splitext(args.out)
plt.savefig(args.out, dpi=160)
plt.savefig(base + ".svg")
print('Wrote', args.out, 'and', base + '.svg')
