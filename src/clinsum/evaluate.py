"""Các chỉ số an toàn dùng cho đề tài (không cần mô hình, chỉ so sánh tập thực thể).

- entity_recall: bao nhiêu thực thể của bản gốc còn xuất hiện trong bản tóm tắt.
- hallucination_rate: tỉ lệ thực thể trong bản tóm tắt KHÔNG có trong bản gốc.

Đầu vào là các tập chuỗi (đã được NER trích ra). Chuẩn hóa: chữ thường, bỏ khoảng trắng thừa.
Khi chưa có NER, có thể dùng tập con dễ trích là các con số (liều, tuổi, xét nghiệm).

- novel_ngram_rate, repetition_rate: đặc trưng của tóm tắt trừu tượng (mức viết lại, lỗi lặp từ).
"""
from typing import Iterable, List, Set, Tuple


def _norm(entities: Iterable[str]) -> Set[str]:
    return {" ".join(e.lower().split()) for e in entities if e and e.strip()}


def entity_recall(source_entities: Iterable[str], summary_entities: Iterable[str]) -> float:
    src, summ = _norm(source_entities), _norm(summary_entities)
    if not src:
        return 1.0
    return len(src & summ) / len(src)


def hallucination_rate(source_entities: Iterable[str], summary_entities: Iterable[str]) -> float:
    src, summ = _norm(source_entities), _norm(summary_entities)
    if not summ:
        return 0.0
    return len(summ - src) / len(summ)


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def novel_ngram_rate(source_tokens: List[str], summary_tokens: List[str], n: int = 2) -> float:
    """Tỉ lệ n-gram của tóm tắt không có trong đầu vào: đo mức độ "trừu tượng" (viết lại)."""
    summ = _ngrams(summary_tokens, n)
    if not summ:
        return 0.0
    src = set(_ngrams(source_tokens, n))
    return sum(g not in src for g in summ) / len(summ)


def repetition_rate(summary_tokens: List[str], n: int = 3) -> float:
    """Tỉ lệ n-gram bị lặp lại trong chính bản tóm tắt (lỗi điển hình của seq2seq không coverage)."""
    grams = _ngrams(summary_tokens, n)
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)
