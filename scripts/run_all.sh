#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from repo root and expose it on PYTHONPATH
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------
# Config (override via env)
# ------------------------
DATASET="${DATASET:-tinyshakespeare}"   # tinyshakespeare | tinystories
STEPS="${STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-192}"
DIM="${DIM:-384}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-6}"
LR="${LR:-3e-4}"
DECODE_STEPS="${DECODE_STEPS:-256}"
VRAM_SEQ="${VRAM_SEQ:-128 256 512 1024 1536 2048}"
LABEL="${LABEL:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr ' ' '_' | tr -d '\r')}"
CKPT="${CKPT:-out/best.pt}"
OUTDIR="${OUTDIR:-out}"
DO_TRAIN="${DO_TRAIN:-auto}"  # auto | 1 | 0

echo "== NanoFalcon run-all =="
echo "DATASET=$DATASET STEPS=$STEPS BATCH=$BATCH_SIZE SEQ=$SEQ_LEN DIM=$DIM L=$LAYERS H=$HEADS LR=$LR"
echo "LABEL=$LABEL CKPT=$CKPT OUTDIR=$OUTDIR"

mkdir -p "$OUTDIR"

# Ensure matplotlib present
python - <<'PY' || pip install -q matplotlib
import importlib; importlib.import_module("matplotlib")
print("matplotlib OK")
PY

# Prepare data
if [[ "$DATASET" == "tinyshakespeare" ]]; then
  python data/prepare_tinyshakespeare.py
else
  python data/prepare_tinystories.py
fi

# Build CUDA extension (safe to re-run)
python setup_cuda.py build_ext --inplace

# Train (optional)
train_needed=0
if [[ "$DO_TRAIN" == "1" ]]; then
  train_needed=1
elif [[ "$DO_TRAIN" == "auto" && ! -f "$CKPT" ]]; then
  train_needed=1
fi

if [[ "$train_needed" == "1" ]]; then
  echo ">> Training..."
  python train.py --data "$DATASET" --steps "$STEPS" --batch_size "$BATCH_SIZE" --seq_len "$SEQ_LEN" \
    --dim "$DIM" --n_layers "$LAYERS" --n_heads "$HEADS" --lr "$LR" --compile --log_csv "$OUTDIR/train_log.csv"
else
  echo ">> Skipping training (CKPT exists or DO_TRAIN=0)"
fi

# Plots: training curve
python scripts/plot_training_curve.py --log "$OUTDIR/train_log.csv" --out "$OUTDIR/fig_training_curve.png" || echo "No train_log.csv; skipping curve."

# Micro-bench RMSNorm
python scripts/bench_rmsnorm.py --iters 200 --dtype fp16 --out "$OUTDIR/rmsnorm_bench.csv"
python scripts/plot_rmsnorm.py --csv "$OUTDIR/rmsnorm_bench.csv" --out "$OUTDIR/fig_rmsnorm.png"

# Decode throughput (append label)
python scripts/bench_decode_tps.py --ckpt "$CKPT" --steps "$DECODE_STEPS" --label "$LABEL" --out "$OUTDIR/decode_bench.csv"
python scripts/plot_tokens_sec.py --csv "$OUTDIR/decode_bench.csv" --out "$OUTDIR/fig_tokens_sec.png"

# KV vs no-KV
python scripts/bench_kv_vs_nokv.py --ckpt "$CKPT" --steps "$DECODE_STEPS" --label "$LABEL" --dtype fp16 --out "$OUTDIR/kv_vs_nokv.csv"
python scripts/plot_kv_vs_nokv.py --csv "$OUTDIR/kv_vs_nokv.csv" --out "$OUTDIR/fig_kv_vs_nokv.png"

# KV vs no-KV across context lengths (line plots)
python scripts/bench_kv_curve.py --ckpt "$CKPT" --steps "$DECODE_STEPS" --dtype fp16 --label "$LABEL" --out "$OUTDIR/kv_curve.csv"
python scripts/plot_kv_curve.py  --csv "$OUTDIR/kv_curve.csv" --out_prefix "$OUTDIR/fig_kv_curve" --logx

# VRAM vs seq_len
python scripts/vram_vs_seq.py --ckpt "$CKPT" --out "$OUTDIR/vram_seq.csv" --seq $VRAM_SEQ
python scripts/plot_vram_seq.py --csv "$OUTDIR/vram_seq.csv" --out "$OUTDIR/fig_vram_seq.png"

# End-to-end ablation (RMSNorm ref vs fused)
python scripts/ablation_end2end.py --ckpt "$CKPT" --steps 256 --dtype fp16 --out "$OUTDIR/ablation_rmsnorm.csv"
python scripts/plot_ablation.py --csv "$OUTDIR/ablation_rmsnorm.csv" --out "$OUTDIR/fig_ablation.png"

echo "== Done. Figures and CSVs in $OUTDIR =="
ls -1 "$OUTDIR"
