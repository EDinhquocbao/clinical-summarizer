"""Tách từ, tách câu đơn giản (không cần thư viện ngoài) dùng chung cho mô hình và baseline.

Giữ nguyên số và dấu chấm thập phân ("2.5", "81") vì liều thuốc là thông tin quan trọng.
"""
import re
from typing import List

TOKEN_RE = re.compile(r"\[DEID\]|\d+(?:\.\d+)?|[A-Za-z]+(?:['\-][A-Za-z]+)*|[^\sA-Za-z\d]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\[])|\n{2,}")
NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")


def tokenize(text: str, lower: bool = True) -> List[str]:
    toks = TOKEN_RE.findall(text)
    return [t.lower() for t in toks] if lower else toks


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_RE.split(text) if s and s.strip()]


def detokenize(tokens: List[str]) -> str:
    out = " ".join(tokens)
    out = re.sub(r" ([.,;:!?)%])", r"\1", out)
    out = re.sub(r"([(]) ", r"\1", out)
    return out


def numbers(tokens: List[str]) -> List[str]:
    return [t for t in tokens if NUMBER_RE.match(t)]
