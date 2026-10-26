"""Sinh tóm tắt cho tập test bằng beam search, ghi ra jsonl {"id", "prediction", "p_gen"}.

Chạy:  python -m clinsum.summarization.predict --checkpoint models/pointer_gen_cov/best.pt --limit 1000
"""
import argparse
import json
import time

import torch

from clinsum.paths import ROOT
from clinsum.text import detokenize, tokenize
from clinsum.summarization.data import read_jsonl
from clinsum.summarization.train import build_model
from clinsum.summarization.vocab import Vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--min_len", type=int, default=None)
    ap.add_argument("--block_ngram", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    vocab = Vocab.load(ROOT / cfg["vocab_path"])
    model = build_model(cfg, vocab).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = read_jsonl(ROOT / cfg["data_dir"] / f"{args.split}.jsonl")[: args.limit]
    out_path = ROOT / (args.out or f"outputs/predictions/{cfg['name']}_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            src = tokenize(r["source"])[: cfg["max_source_len"]]
            src_ext, oovs = vocab.encode_source_ext(src)
            ids, p_gen = model.beam_search(
                torch.tensor([vocab.encode(src)], device=device),
                torch.tensor([src_ext], device=device),
                len(oovs), vocab.bos, vocab.eos, beam_size=args.beam,
                max_len=args.max_len or cfg.get("decode_max_len", 120),
                min_len=args.min_len or cfg.get("decode_min_len", 20),
                block_ngram=args.block_ngram,
            )
            pred = detokenize(vocab.decode_ext(ids, oovs))
            f.write(json.dumps({"id": r["id"], "prediction": pred, "p_gen": p_gen}, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)
    print("Đã ghi", out_path)


if __name__ == "__main__":
    main()
