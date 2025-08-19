import argparse, torch
from model import TinyLM, build_sincos, prealloc_kvcache
from tokenizers import Tokenizer

def generate(model, tok, prompt, max_new_tokens=128, temperature=1.0):
    device = next(model.parameters()).device
    sin, cos = build_sincos(8192, model.dim // model.n_heads, device)
    ids = torch.tensor(tok.encode(prompt).ids, device=device).unsqueeze(0)
    cache = prealloc_kvcache(1, ids.size(1)+max_new_tokens, model.n_heads, model.dim//model.n_heads, device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        for t in range(max_new_tokens):
            logits = model(ids[:, -1:].contiguous(), sin, cos, cache, start_pos=ids.size(1)-1)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--prompt', type=str, default='Once upon a time')
    ap.add_argument('--max_new_tokens', type=int, default=128)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    tok = Tokenizer.from_str(ckpt['tok'])
    model = TinyLM(tok.get_vocab_size()).cuda().eval()
    model.load_state_dict(ckpt['model'])
    txt = generate(model, tok, args.prompt, args.max_new_tokens)
    print(txt)

if __name__ == '__main__':
    main()