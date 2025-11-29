# Entry 001: Pipeline Refactoring and Streaming Data

**Date:** 2025-11-29

---

Started around midnight. Coffee in hand. Goal: get the training pipeline working on the combined TinyShakespeare + TinyStories dataset.

---

### The "God File" Problem

`train.py` had grown into a monster. 600+ lines of spaghetti mixing data loading, checkpointing, metrics, LR scheduling... you name it. Every time I wanted to change something, I had to scroll through the whole thing.

Spent a couple hours extracting everything into `tinylm/training/`:
- `data.py` — datasets and tokenizer
- `checkpoint.py` — save/load with rotation
- `metrics.py` — TensorBoard wrapper
- `scheduler.py` — warmup + cosine decay, early stopping
- `utils.py` — signal handling, evaluation

Added 17 tests. Feels cleaner now. `train.py` is down to ~380 lines.

---

### The Memory Problem

Tried to run on the combined dataset. TinyStories alone is 1.9GB, 2 million stories. First attempt: OOM.

The issue was twofold:
1. HuggingFace was downloading the entire dataset into RAM
2. My `CharDataset` was tokenizing the whole file upfront

Fix: streaming everywhere.

For download, just add `streaming=True` to `load_dataset()`. For training, wrote a `StreamingDataset` that reads the file in 1MB chunks and yields sequences as a generator. Never loads more than a few MB into memory.

Works now. The 1.9GB file loads fine.

---

### The "Stuck at 0%" Mystery

Kicked off training. Progress bar showed 0%. Waited. And waited. Five minutes later, still 0%.

Ctrl+C didn't work properly either — it set a flag but the code was stuck in a loop that never checked it.

Turns out: the initial validation was iterating through 4.6 MILLION sequences. At 32 per batch, that's 145,000 batches. No wonder it took forever.

Quick fix: added `max_eval_batches=100` to sample instead of evaluating everything. Validation now takes 4 seconds instead of... I don't even want to know how long.

Also added shutdown checks inside the evaluation loop so Ctrl+C actually works now.

---

### First Real Training Run

Finally, actual training:

```
data=combined, batch_size=32, seq_len=256, steps=10000
grad_accum=2, mixed_precision=true, early_stopping_patience=5
```

The A2000 was humming along at ~8 iterations/sec. Not bad for a 6GB card.

**Results after ~10 minutes:**
- Training perplexity: ~10-12 (nice, model is learning)
- Validation perplexity: ~600 (uh oh)
- Early stopping triggered at step 2100

That's a 60x gap between train and val. Classic overfitting.

The model is memorizing the training data but not generalizing at all. Val loss plateaued around step 1600 and just... stopped improving.

---

### What's Going On?

A few hypotheses:

1. **Dataset mismatch** — Shakespeare (old English, poetic) mixed with TinyStories (simple children's stories) might confuse the model. Very different distributions.

2. **Model too small** — 13M params trying to learn 1.9GB of diverse text. Might not have enough capacity to generalize.

3. **Model too big for the task** — Or maybe the opposite? With dropout=0.1, it might be memorizing instead of learning patterns.

4. **Learning rate too high** — 3e-4 might be overshooting good minima. The train loss keeps dropping but val doesn't follow.

5. **Early stopping too aggressive** — Patience=5 means we stop after 500 steps without improvement. Maybe val needs more time to catch up.

---

### Experiments to Try

**Sanity check first:** Run on just TinyShakespeare. It's small (~1MB), homogeneous. Should get val PPL around 10-20. If this fails, something is fundamentally wrong.

```bash
uv run python train.py data=tinyshakespeare training.steps=5000
```

**If Shakespeare works, try:**

1. **More regularization** — Crank up dropout to 0.2, weight_decay to 0.1
2. **Lower LR** — Try 1e-4 instead of 3e-4, longer warmup
3. **Bigger model** — medium config (dim=512, 8 layers)
4. **Disable early stopping** — Let it run longer, see if val eventually improves
5. **Single dataset** — Just TinyStories, no Shakespeare mixing

---

### Notes for Tomorrow

- Tokenizer building still takes 2.5 minutes on the combined dataset. The streaming helps but it's still slow. Maybe cache the tokenizer?

- Memory usage is fine. Only using ~3-4GB of the 8GB VRAM. Could probably increase batch size.

- Need to check if the validation set is representative. Maybe the train/val split is weird?

- Consider curriculum learning: train on Shakespeare first (smaller, cleaner), then fine-tune on TinyStories.

---

### Current State

Pipeline is solid:
- Streaming data loading ✓
- Checkpointing with rotation ✓
- TensorBoard logging ✓
- Early stopping ✓
- Graceful shutdown ✓

Model is learning but not generalizing. Need to figure out why.

Time to sleep. Will run the Shakespeare sanity check tomorrow.

---

*"The first step in fixing a problem is admitting you have one. The second step is adding more dropout."* — Ancient ML Proverb
