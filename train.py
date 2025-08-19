from torch.optim import AdamW
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
    trainer = BpeTrainer(vocab_size=4096, special_tokens=["<unk>"])
    tok.train(files=corpus_paths, trainer=trainer)
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
    ap.add_argument('--flash', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == 'tinystories':
        train_path = 'data/tinystories_train.txt'
        val_path   = 'data/tinystories_val.txt'
    else:
        train_path = 'data/tinyshakespeare_train.txt'
        val_path   = 'data/tinyshakespeare_val.txt'

    if not os.path.exists('tokenizer.json'):
        build_tokenizer([train_path, val_path], 'tokenizer.json')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('tokenizer.json')

    with open(train_path, 'r') as f: train_text = f.read()
    with open(val_path, 'r') as f: val_text = f.read()

    train_ds = CharDataset(train_text, args.seq_len, tok)
    val_ds   = CharDataset(val_text, args.seq_len, tok)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TinyLM(vocab_size=tok.get_vocab_size(), dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=args.lr)

    sin, cos = build_sincos(4096, model.dim // model.n_heads, device)

    best = 1e9
    os.makedirs('out', exist_ok=True)

    for step, (x, y) in enumerate(tqdm(train_dl, total=args.steps)):
        x, y = x.to(device), y.to(device)
        logits = model(x, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            val_loss = evaluate(model, val_dl, sin, cos, device)
            if val_loss < best:
                best = val_loss
                torch.save({'model': model.state_dict(), 'tok': tok.to_str()}, 'out/best.pt')
        if step+1 >= args.steps:
            break

if __name__ == '__main__':
    main()