"""Tiền xử lý discharge summary.

Hai việc đã cài sẵn:
1. normalize_placeholders: thống nhất ký hiệu khử định danh.
   - MIMIC-III dùng dạng [**Tên loại**]
   - MIMIC-IV-Note dùng dạng ___ (gạch dưới liên tiếp)
2. split_sections: tách văn bản theo các mục có tiêu đề riêng một dòng,
   ví dụ "Brief Hospital Course:".

Giới hạn: tiêu đề nằm cùng dòng với nội dung sẽ chưa được tách.
Hãy kiểm tra lại trên dữ liệu thật rồi chỉnh regex nếu cần.
"""
import re
from typing import Dict

MIMIC3_PLACEHOLDER = re.compile(r"\[\*\*.*?\*\*\]")
MIMIC4_PLACEHOLDER = re.compile(r"_{3,}")
SECTION_HEADER = re.compile(r"^([A-Za-z][A-Za-z ,/&()\-]{2,60}):[ \t]*$", re.MULTILINE)


def normalize_placeholders(text: str, token: str = "[DEID]") -> str:
    text = MIMIC3_PLACEHOLDER.sub(token, text)
    text = MIMIC4_PLACEHOLDER.sub(token, text)
    return text


def split_sections(text: str) -> Dict[str, str]:
    """Trả về {tên_mục_viết_thường: nội_dung}. Phần trước mục đầu tiên gộp vào 'preamble'."""
    matches = list(SECTION_HEADER.finditer(text))
    sections: Dict[str, str] = {}
    if not matches:
        return {"preamble": text.strip()}
    pre = text[: matches[0].start()].strip()
    if pre:
        sections["preamble"] = pre
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = m.group(1).strip().lower()
        body = text[m.end():end].strip()
        if name in sections:  # tiêu đề lặp lại: nối tiếp
            sections[name] += "\n" + body
        else:
            sections[name] = body
    return sections
