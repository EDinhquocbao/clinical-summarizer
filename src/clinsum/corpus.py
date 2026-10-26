"""Tạo cặp (input, tóm tắt) từ discharge summary: bài toán "hospital course summarization".

- Đích (target): mục Hospital Course / Brief Hospital Course.
- Đầu vào (source): các mục còn lại của chính ghi chú đó (bỏ mục đích để không lộ đáp án).

Dùng được cho cả dữ liệu tổng hợp (Asclepius) lẫn MIMIC sau này, chỉ khác nguồn văn bản.
Chia train/val/test theo bệnh nhân để tránh rò rỉ dữ liệu.
"""
import random
from typing import Dict, Iterable, List, Optional, Tuple

from clinsum.preprocess import normalize_placeholders, split_sections

TARGET_SECTIONS = ("brief hospital course", "hospital course", "summary of hospital course")
# Các mục chỉ chứa tóm tắt/kết luận, bỏ khỏi đầu vào để mô hình không chép sẵn câu trả lời
DROP_FROM_SOURCE = ("discharge diagnosis", "discharge instructions", "summary")


def make_pair(note: str, min_target_words: int = 20) -> Optional[Tuple[str, str]]:
    sections = split_sections(normalize_placeholders(note))
    target_key = next((k for k in TARGET_SECTIONS if k in sections), None)
    if target_key is None:
        return None
    target = sections[target_key]
    source_parts = []
    for name, body in sections.items():
        if name == target_key or name in TARGET_SECTIONS or name in DROP_FROM_SOURCE or not body:
            continue
        source_parts.append(body if name == "preamble" else f"{name}: {body}")
    source = "\n".join(source_parts)
    if len(target.split()) < min_target_words or len(source.split()) < min_target_words:
        return None
    return source, target


def split_by_group(
    rows: Iterable[Dict], key: str, ratios=(0.8, 0.1, 0.1), seed: int = 42
) -> Dict[str, List[Dict]]:
    rows = list(rows)
    groups = sorted({r[key] for r in rows})
    random.Random(seed).shuffle(groups)
    n_train = int(len(groups) * ratios[0])
    n_val = int(len(groups) * ratios[1])
    which = {g: "train" for g in groups[:n_train]}
    which.update({g: "val" for g in groups[n_train:n_train + n_val]})
    which.update({g: "test" for g in groups[n_train + n_val:]})
    out: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[which[r[key]]].append(r)
    return out
