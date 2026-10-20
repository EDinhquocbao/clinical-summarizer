"""Các chỉ số an toàn dùng cho đề tài (không cần mô hình, chỉ so sánh tập thực thể).

- entity_recall: bao nhiêu thực thể của bản gốc còn xuất hiện trong bản tóm tắt.
- hallucination_rate: tỉ lệ thực thể trong bản tóm tắt KHÔNG có trong bản gốc.

Đầu vào là các tập chuỗi (đã được NER trích ra). Chuẩn hóa: chữ thường, bỏ khoảng trắng thừa.
"""
from typing import Iterable, Set


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
