"""Prepare TinyStories dataset with streaming (memory efficient)."""

import os

from datasets import load_dataset

os.makedirs('data', exist_ok=True)

# Stream train split - never loads full dataset into memory
print("Streaming TinyStories train split...")
train_ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
count = 0
with open('data/tinystories_train.txt', 'w') as f:
    for example in train_ds:
        f.write(example['text'] + '\n')
        count += 1
        if count % 100000 == 0:
            print(f"  Written {count:,} stories...")
print(f"Written {count:,} train stories")

# Stream validation split
print("Streaming TinyStories validation split...")
val_ds = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
count = 0
with open('data/tinystories_val.txt', 'w') as f:
    for example in val_ds:
        f.write(example['text'] + '\n')
        count += 1
print(f"Written {count:,} val stories")

print('Done! TinyStories train/val written to data/')
