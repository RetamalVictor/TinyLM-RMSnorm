# Entry 002: The Tokenizer Trap

**Date:** 2025-11-29

---

Woke up to check the overnight experiments. Four runs completed. Results were... disappointing. But then I found something interesting.

## The Overnight Results

Ran four experiments on the combined TinyShakespeare + TinyStories dataset:

| Experiment | Config | Best Val PPL | Steps | Notes |
|------------|--------|--------------|-------|-------|
| Exp 1: Regularization | dropout=0.2, wd=0.1 | 396 | 1,100 | Early stopped |
| Exp 2: Medium Model | 29M params | 376 | 1,100 | Early stopped |
| Exp 3: Low LR | lr=1e-4, patience=20 | ~1,000 | partial | Very slow |
| Exp 4: Long Run | lr=1e-4, patience=250 | 388 | 100,000 | Full run |

None cracked PPL < 300. More regularization didn't help. Bigger model didn't help. Lower learning rate didn't help. 100k steps didn't help.

Something deeper was wrong.

## The Sanity Check That Worked

Before the overnight runs, I did a quick test on TinyShakespeare alone:

```
Best val PPL: 112
Train PPL: ~31
Gap: 3.5x
```

Not amazing, but reasonable. And the inference output actually sounded like Shakespeare:

```
To be or not to be made me to be gone, To find ing to make less
than this hour hath he would fain's of men...
KING RICHARD III: For shame, that he re ction shall we have been...
```

Wait. "find ing"? "re ction"? Those spaces shouldn't be there.

## The Symptom

Ran inference on the best overnight model (100k steps, PPL 388):

```
Once upon a time and then went on ions and saw the t weet ountain
º ( avail ny ventually ings you th orn sses ill as many times.
She is le cture s igh ing ly lifted off ic ke en coun tered...
```

Word salad with broken words. "t weet"? "igh ing ly"? "le cture"?

At first I thought: "The model just hasn't learned language structure." But 100k steps should produce *something* coherent. This looked like a different kind of failure.

## The Investigation

Tested the tokenizer directly:

```python
>>> tok.encode("Once upon a time there was a little girl").tokens
['Once', 'upon', 'a', 'time', 'there', 'was', 'a', 'little', 'girl']

>>> tok.decode(tok.encode("carefully built mountain").ids)
'carefully built mountain'
```

Encoding and decoding works perfectly for normal text. The tokenizer itself is fine.

Then I checked the vocabulary:

```python
>>> tok.get_vocab()['igh']
753
>>> tok.get_vocab()['ing']
191
>>> tok.get_vocab()['ly']
239
>>> tok.get_vocab()['ed']
174
```

Found it.

## The Root Cause

The BPE tokenizer learned common suffixes as standalone tokens: `igh`, `ing`, `ly`, `ed`, `ful`. These are valid tokens in the vocabulary.

When the model generates a sequence like `[token_for_word, 753, 191, 239]`, the decoder outputs them with spaces: `"word igh ing ly"` instead of `"wordighingly"`.

The problem is the tokenizer architecture:
- We use `Whitespace` pre-tokenizer, which splits on whitespace first
- BPE then learns subword patterns *within* words
- But on decode, each token is treated as a separate word
- There's no mechanism to merge subwords back together

GPT-2 uses `ByteLevel` pre-tokenizer which handles this properly — subword tokens can merge on decode because the encoding preserves the information about word boundaries.

## Why This Matters

The model was actually learning. It learned that `ing` and `ly` are common patterns. It learned sentence structure. It learned to generate contextually relevant tokens.

But it couldn't express coherent text because the tokenizer fragmented its outputs. The high validation perplexity wasn't because the model was bad — it was because the model's predictions looked like garbage when decoded.

All those hours tuning hyperparameters (dropout, LR, model size, patience) were wasted. The fundamental issue was tokenizer architecture.

## The Deeper Lesson

I spent hours in "model tuning" mode when I should have been in "pipeline validation" mode. Classic ML debugging mistake.

The workflow should have been:
1. Train tiny model for 100 steps
2. Run inference immediately
3. **Look at the actual output** — not just the loss
4. If output is broken, fix data pipeline before scaling up

Instead, I trusted the loss numbers and launched overnight experiments. The loss was measuring something real (next-token prediction), but the decoded output was meaningless.

## The Fix

Options:

1. **Switch to ByteLevel pre-tokenizer** (GPT-2 style)
   - Handles subword merging properly
   - Requires retraining tokenizer and model from scratch

2. **Increase vocab size to 8k-16k**
   - More complete words, fewer subword fragments
   - Might reduce but won't eliminate the issue

3. **Add a BPE Decoder with merge rules**
   - The `tokenizers` library supports this
   - Could be a quick fix for existing tokenizer

4. **Use character-level tokenizer**
   - No subword issues at all
   - But much longer sequences, slower training

Going with option 1. It's the most robust fix, and we need to retrain anyway since all our models learned a broken representation.

---

*"Always look at your model's output before tuning hyperparameters. Loss is a proxy. Text is truth."*
