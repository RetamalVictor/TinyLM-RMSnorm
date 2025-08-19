import os, datasets
os.makedirs('data', exist_ok=True)
ds = datasets.load_dataset('roneneldan/TinyStories', split='train')
val = datasets.load_dataset('roneneldan/TinyStories', split='validation')
with open('data/tinystories_train.txt','w') as f:
    for r in ds['text']: f.write(r + '\n')
with open('data/tinystories_val.txt','w') as f:
    for r in val['text']: f.write(r + '\n')
print('Wrote TinyStories train/val.')