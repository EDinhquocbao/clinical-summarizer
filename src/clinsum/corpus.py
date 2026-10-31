"""Tạo cặp (input, tóm tắt) từ discharge summary: bài toán "hospital course summarization".

- Đích (target): mục Hospital Course / Brief Hospital Course.
- Đầu vào (source): các mục còn lại của chính ghi chú đó (bỏ mục đích để không lộ đáp án).

Dùng được cho cả dữ liệu tổng hợp (Asclepius) lẫn MIMIC sau này, chỉ khác nguồn văn bản.
Chia train/val/test theo bệnh nhân để tránh rò rỉ dữ liệu.
"""
import hashlib
import random
import re
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


# ----------------------------------------------------------------------------- MIMIC-IV-Note
# Tiêu đề cấp cao nhất của mẫu discharge summary MIMIC-IV (theo mô tả công khai của bộ dữ liệu).
# Chỉ tách ở các tiêu đề này; tiêu đề con viết hoa bên trong một mục (ADMISSION LABS:,
# TRANSITIONAL ISSUES:...) vẫn thuộc mục cha, để mục Brief Hospital Course không bị cắt cụt.
MIMIC_HEADERS = {
    "name": r"name",
    "unit no": r"unit no",
    "admission date": r"admission date",
    "discharge date": r"discharge date",
    "date of birth": r"date of birth",
    "sex": r"sex",
    "service": r"service",
    "allergies": r"allergies",
    "attending": r"attending",
    "chief complaint": r"chief complaint",
    "major surgical or invasive procedure": r"major surgical (?:or|and) invasive procedures?",
    "history of present illness": r"history of (?:the )?present illness",
    "past medical history": r"past medical history",
    "social history": r"social history",
    "family history": r"family history",
    "physical exam": r"physical exam(?:ination)?",
    "pertinent results": r"pertinent results",
    "brief hospital course": r"(?:brief )?hospital course",
    "medications on admission": r"medications on admission",
    "discharge medications": r"discharge medications",
    "discharge disposition": r"discharge disposition",
    "facility": r"facility",
    "discharge diagnosis": r"discharge diagnos[ie]s",
    "discharge condition": r"discharge condition",
    "discharge instructions": r"discharge instructions",
    "followup instructions": r"follow[- ]?up instructions",
}
_MIMIC_HEADER_RE = re.compile(
    r"^[ \t]*(" + "|".join(f"(?P<h{i}>{p})" for i, p in enumerate(MIMIC_HEADERS.values())) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_MIMIC_KEYS = list(MIMIC_HEADERS)

# Mục đưa vào đầu vào, theo thứ tự ưu tiên (mô hình chỉ đọc ~512 token đầu nên mục quan trọng đặt trước).
# Không đưa: mục đích (BHC), thuốc, hướng dẫn xuất viện (thường nhắc lại diễn tiến điều trị → lộ đáp án),
# thông tin hành chính.
MIMIC_SOURCE_SECTIONS = (
    "chief complaint",
    "history of present illness",
    "major surgical or invasive procedure",
    "past medical history",
    "pertinent results",
    "physical exam",
)


def split_mimic_sections(text: str) -> Dict[str, str]:
    """Tách discharge summary MIMIC-IV theo MIMIC_HEADERS. Nội dung có thể nằm cùng dòng với tiêu đề."""
    matches = list(_MIMIC_HEADER_RE.finditer(text))
    sections: Dict[str, str] = {}
    if not matches:
        return {"preamble": text.strip()} if text.strip() else {}
    if text[: matches[0].start()].strip():
        sections["preamble"] = text[: matches[0].start()].strip()
    for i, m in enumerate(matches):
        key = next(_MIMIC_KEYS[int(g[1:])] for g, v in m.groupdict().items() if v is not None)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        sections[key] = (sections[key] + "\n" + body).strip() if key in sections else body
    return sections


def _cap_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def make_mimic_pair(
    note: str,
    source_sections=MIMIC_SOURCE_SECTIONS,
    min_target_words: int = 30,
    min_source_words: int = 50,
    section_max_words: int = 0,
) -> Tuple[Optional[Tuple[str, str]], Dict[str, str]]:
    """Trả về ((source, target) hoặc None, các mục đã tách). Mục đích: Brief Hospital Course."""
    sections = split_mimic_sections(normalize_placeholders(note))
    target = sections.get("brief hospital course", "")
    parts = [f"{name}: {_cap_words(sections[name], section_max_words)}"
             for name in source_sections if sections.get(name)]
    source = "\n".join(parts)
    if len(target.split()) < min_target_words or len(source.split()) < min_source_words:
        return None, sections
    return (source, target), sections


def subject_split(subject_id, ratios=(0.8, 0.1, 0.1), seed: int = 42) -> str:
    """Chia theo bệnh nhân bằng băm cố định: cùng subject_id luôn vào cùng một tập, không cần nạp hết dữ liệu."""
    h = int(hashlib.md5(f"{seed}:{subject_id}".encode()).hexdigest(), 16) % 10_000 / 10_000
    return "train" if h < ratios[0] else ("val" if h < ratios[0] + ratios[1] else "test")


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
