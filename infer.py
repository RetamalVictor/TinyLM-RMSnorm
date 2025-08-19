import argparse, torch, random
from model import TinyLM, build_sincos, prealloc_kvcache
from tokenizers import Tokenizer

def sample_top_p(logits, top_p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > top_p
    # Keep at least 1 token
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_idx.gather(-1, idx)
    return next_token

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=128, temperature=1.0, top_p=0.9,
             repetition_penalty=1.1, freq_penalty=0.0, presence_penalty=0.0, seed=0, stream=False):
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    sin, cos = build_sincos(8192, model.dim // model.n_heads, device)
    ids = torch.tensor(tok.encode(prompt).ids, device=device).unsqueeze(0)
    cache = prealloc_kvcache(1, ids.size(1)+max_new_tokens, model.n_heads, model.dim//model.n_heads, device, dtype=next(model.parameters()).dtype)

    recent_window = 512
    for _ in range(max_new_tokens):
        logits = model(ids[:, -1:], sin, cos, cache, start_pos=ids.size(1)-1)
        logits = logits[:, -1, :]
        # Repetition penalty / frequency & presence penalties
        if ids.size(1) > 0 and (repetition_penalty > 1.0 or freq_penalty > 0.0 or presence_penalty > 0.0):
            recent = ids[:, -recent_window:]
            for b in range(ids.size(0)):
                unique, counts = torch.unique(recent[b], return_counts=True)
                if repetition_penalty > 1.0:
                    logits[b, unique] /= repetition_penalty
                if freq_penalty > 0.0:
                    logits[b, unique] -= freq_penalty * counts.to(logits.dtype)
                if presence_penalty > 0.0:
                    logits[b, unique] -= presence_penalty
        # Temperature
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        # Nucleus sampling
        next_id = sample_top_p(logits, top_p=top_p)
        ids = torch.cat([ids, next_id], dim=1)
        if stream:
            print(tok.decode(ids[0].tolist()), flush=True)
    return tok.decode(ids[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--prompt', type=str, default='Once upon a time')
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--repetition_penalty', type=float, default=1.1)
    ap.add_argument('--freq_penalty', type=float, default=0.0)
    ap.add_argument('--presence_penalty', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stream', action='store_true')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    tok = Tokenizer.from_str(ckpt['tok'])

    cfg = ckpt.get('config', None)
    if cfg is None:
        cfg = {'dim': 384, 'n_layers': 6, 'n_heads': 6, 'vocab_size': tok.get_vocab_size()}

    model = TinyLM(vocab_size=cfg['vocab_size'], dim=cfg['dim'], n_layers=cfg['n_layers'], n_heads=cfg['n_heads']).cuda().eval()

    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    txt = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_p,
                   args.repetition_penalty, args.freq_penalty, args.presence_penalty, args.seed, args.stream)
    print(txt)

if __name__ == "__main__":
    main()
