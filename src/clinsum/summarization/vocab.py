"""Từ điển và ánh xạ "từ điển mở rộng" cho cơ chế copy (See et al., 2017).

Từ ngoài từ điển (OOV) trong đầu vào được gán chỉ số tạm vocab_size + i, nhờ đó mô hình
Pointer-Generator vẫn chép được tên thuốc hiếm, số liệu... từ văn bản gốc.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
SPECIALS = [PAD, UNK, BOS, EOS]


class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad, self.unk, self.bos, self.eos = (self.stoi[s] for s in SPECIALS)

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, token_lists: Iterable[List[str]], max_size: int, min_freq: int = 2) -> "Vocab":
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        words = [w for w, c in counter.most_common(max_size - len(SPECIALS)) if c >= min_freq]
        return cls(SPECIALS + words)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk) for t in tokens]

    def encode_source_ext(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        """Chỉ số của đầu vào trong từ điển mở rộng và danh sách OOV của văn bản này."""
        ids, oovs = [], []
        for t in tokens:
            i = self.stoi.get(t)
            if i is None:
                if t not in oovs:
                    oovs.append(t)
                i = len(self.itos) + oovs.index(t)
            ids.append(i)
        return ids, oovs

    def encode_target_ext(self, tokens: List[str], oovs: List[str]) -> List[int]:
        ids = []
        for t in tokens:
            i = self.stoi.get(t)
            if i is None:
                i = len(self.itos) + oovs.index(t) if t in oovs else self.unk
            ids.append(i)
        return ids

    def decode_ext(self, ids: List[int], oovs: List[str]) -> List[str]:
        n = len(self.itos)
        return [self.itos[i] if i < n else (oovs[i - n] if i - n < len(oovs) else UNK) for i in ids]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
