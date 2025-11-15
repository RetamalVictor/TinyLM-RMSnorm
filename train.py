import argparse, os, csv
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from model import TinyLM, build_sincos

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tok = tokenizer
        self.ids = self.tok.encode(text).ids
    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)
    def __getitem__(self, i):
        x = self.ids[i:i+self.seq_len]
        y = self.ids[i+1:i+self.seq_len+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_tokenizer(corpus_paths, out_path):
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<unk>"])
    # Streamed training to avoid RAM blowups
    def line_iter():
        for p in corpus_paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()
    tok.train_from_iterator(line_iter(), trainer=trainer)
    tok.save(out_path)
    return tok

@torch.no_grad()
def evaluate(model, dl, sin, cos, device):
    model.eval()
    loss_sum = 0
    n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss_sum += loss.item(); n += 1
    model.train()
    return loss_sum / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='tinystories')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seq_len', type=int, default=256)
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--log_csv', type=str, default='out/train_log.csv')
    ap.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    ap.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
    ap.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'linear', 'constant'])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == 'tinystories':
        train_path = 'data/tinystories_train.txt'
        val_path   = 'data/tinystories_val.txt'
    else:
        train_path = 'data/tinyshakespeare_train.txt'
        val_path   = 'data/tinyshakespeare_val.txt'

    # Check if data files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please run 'python data/prepare_{args.data}.py' first."
        )
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            f"Please run 'python data/prepare_{args.data}.py' first."
        )

    os.makedirs('out', exist_ok=True)

    # Build or load tokenizer
    try:
        if not os.path.exists('tokenizer.json'):
            print("Building tokenizer...")
            build_tokenizer([train_path, val_path], 'tokenizer.json')
        tok = Tokenizer.from_file('tokenizer.json')
    except Exception as e:
        raise RuntimeError(f"Failed to build/load tokenizer: {e}")

    # Load data files
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        with open(val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read data files: {e}")

    train_ds = CharDataset(train_text, args.seq_len, tok)
    val_ds   = CharDataset(val_text, args.seq_len, tok)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TinyLM(vocab_size=tok.get_vocab_size(), dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr)
    sin, cos = build_sincos(4096, model.dim // model.n_heads, device)

    # Create learning rate scheduler
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.1
        )
    elif args.lr_schedule == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps
        )
    else:  # constant
        scheduler = None

    best = 1e9

    # Helper function to get current learning rate
    def get_lr():
        return opt.param_groups[0]['lr']

    # CSV logger
    with open(args.log_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['step','train_loss','val_loss','lr'])

        step = 0
        train_iter = iter(train_dl)
        pbar = tqdm(total=args.steps)
        while step < args.steps:
            try:
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dl)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)

                # Forward pass with OOM handling
                logits = model(x, sin, cos)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                opt.step()

                # Update learning rate
                if scheduler is not None:
                    scheduler.step()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n[Warning] OOM at step {step}. Clearing cache and skipping batch.")
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            val_loss = ''
            if step % 100 == 0:
                val_loss = evaluate(model, val_dl, sin, cos, device)
                if val_loss < best:
                    best = val_loss
                    base = getattr(model, "_orig_mod", model)
                    torch.save({
                        'model': base.state_dict(),
                        'tok': tok.to_str(),
                        'config': {
                            'dim': base.dim,
                            'n_layers': len(base.blocks),
                            'n_heads': base.n_heads,
                            'vocab_size': tok.get_vocab_size(),
                        }
                    }, 'out/best.pt')
            writer.writerow([step, float(loss.item()), ('' if val_loss=='' else float(val_loss)), get_lr()])
            step += 1
            pbar.set_description(f'Loss: {loss.item():.3f}, LR: {get_lr():.2e}')
            pbar.update(1)
        pbar.close()

if __name__ == '__main__':
    main()