#!/bin/bash
# Overnight experiments for TinyLM on combined dataset
# Run with: bash scripts/overnight_experiments.sh
# Expected duration: ~6-7 hours

set -e  # Exit on error

echo "============================================"
echo "TinyLM Overnight Experiments"
echo "Started: $(date)"
echo "============================================"

# Experiment 1: More regularization
echo ""
echo "[$(date)] Starting Experiment 1: More Regularization"
echo "  - dropout=0.2, weight_decay=0.1"
echo "  - steps=20000"
echo "============================================"
uv run python train.py \
  data=combined \
  model.dropout=0.2 \
  training.weight_decay=0.1 \
  training.steps=20000 \
  training.early_stopping_patience=10 \
  experiment.name=exp1_regularization

echo "[$(date)] Experiment 1 complete"

# Experiment 2: Larger model
echo ""
echo "[$(date)] Starting Experiment 2: Medium Model"
echo "  - model=medium (dim=512, 8 layers)"
echo "  - batch_size=16 (fits in 8GB)"
echo "  - steps=20000"
echo "============================================"
uv run python train.py \
  data=combined \
  model=medium \
  training.steps=20000 \
  training.batch_size=16 \
  training.early_stopping_patience=10 \
  experiment.name=exp2_medium_model

echo "[$(date)] Experiment 2 complete"

# Experiment 3: Lower LR, long run
echo ""
echo "[$(date)] Starting Experiment 3: Low LR Long Run"
echo "  - lr=1e-4 (3x lower)"
echo "  - warmup=1000 steps"
echo "  - steps=100000 (will take ~4 hours)"
echo "============================================"
uv run python train.py \
  data=combined \
  training.lr=1e-4 \
  training.warmup_steps=1000 \
  training.steps=100000 \
  training.early_stopping_patience=250 \
  experiment.name=exp4_low_lr_long_long_patience

echo "[$(date)] Experiment 3 complete"

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Finished: $(date)"
echo "Check TensorBoard: uv run tensorboard --logdir=outputs"
echo "============================================"
