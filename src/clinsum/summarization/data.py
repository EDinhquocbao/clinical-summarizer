"""Dataset và collate cho mô hình seq2seq / Pointer-Generator.

Mỗi dòng jsonl: {"id", "group", "source", "target"}. Văn bản được tách từ bằng clinsum.text.tokenize,
cắt còn max_source_len / max_target_len token (như See et al. cắt 400 / 100).
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from clinsum.text import tokenize
from clinsum.summarization.vocab import Vocab


def read_jsonl(path) -> List[Dict]:
    with open(Path(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class SummDataset(Dataset):
    def __init__(self, rows: List[Dict], vocab: Vocab, max_source_len: int, max_target_len: int):
        self.rows, self.vocab = rows, vocab
        self.max_src, self.max_tgt = max_source_len, max_target_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        r = self.rows[i]
        src = tokenize(r["source"])[: self.max_src]
        tgt = tokenize(r["target"])[: self.max_tgt]
        v = self.vocab
        src_ext, oovs = v.encode_source_ext(src)
        tgt_ext = v.encode_target_ext(tgt, oovs)
        return {
            "id": r["id"],
            "src": v.encode(src),
            "src_ext": src_ext,
            "oovs": oovs,
            # đầu vào decoder dùng từ điển thường (OOV -> UNK), đích dùng từ điển mở rộng
            "dec_in": [v.bos] + v.encode(tgt),
            "dec_out": tgt_ext + [v.eos],
            "dec_out_plain": v.encode(tgt) + [v.eos],
        }


def _pad(seqs: List[List[int]], value: int) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs], dtype=torch.long)


def collate(batch: List[Dict], pad_id: int) -> Dict:
    return {
        "ids": [b["id"] for b in batch],
        "src": _pad([b["src"] for b in batch], pad_id),
        "src_ext": _pad([b["src_ext"] for b in batch], pad_id),
        "src_len": torch.tensor([len(b["src"]) for b in batch]),
        "oovs": [b["oovs"] for b in batch],
        "max_oovs": max(len(b["oovs"]) for b in batch),
        "dec_in": _pad([b["dec_in"] for b in batch], pad_id),
        "dec_out": _pad([b["dec_out"] for b in batch], pad_id),
        "dec_out_plain": _pad([b["dec_out_plain"] for b in batch], pad_id),
    }
