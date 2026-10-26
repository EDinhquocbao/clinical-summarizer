"""Tạo dữ liệu tóm tắt từ Asclepius (discharge summary TỔNG HỢP, công khai, CC-BY-NC-SA 4.0).

Dùng tạm trong lúc chờ quyền truy cập MIMIC. Nguồn: huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes
(Kweon et al., 2023: ghi chú sinh bởi GPT-3.5 từ ca bệnh PMC-Patients; KHÔNG chứa dữ liệu bệnh nhân thật).

Hai chế độ:
- summarization (mặc định): các dòng task == "Summarization".
  source = yêu cầu tóm tắt + ghi chú, target = bản tóm tắt.
- hospital_course: source = các mục khác của ghi chú, target = mục Hospital Course
  (cùng cách sẽ dùng cho MIMIC; trên Asclepius mục này dài hơn phần còn lại nên chỉ để thử).

Chạy:  python scripts/prepare_asclepius.py  [--mode hospital_course]
"""
import argparse
import json
import re

import pandas as pd

from clinsum.corpus import make_pair, split_by_group
from clinsum.paths import EXTERNAL_DIR, PROCESSED_DIR


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", str(text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EXTERNAL_DIR / "asclepius" / "synthetic.csv"))
    ap.add_argument("--mode", choices=["summarization", "hospital_course"], default="summarization")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = []
    if args.mode == "summarization":
        for r in df[df.task == "Summarization"].itertuples():
            rows.append({"id": str(r.patient_id), "group": str(r.patient_id),
                         "source": f"request: {clean(r.question)}\n{clean(r.note)}", "target": clean(r.answer)})
        out_dir = PROCESSED_DIR / "asclepius_sum"
    else:
        for r in df.itertuples():
            pair = make_pair(clean(r.note))
            if pair:
                rows.append({"id": str(r.patient_id), "group": str(r.patient_id),
                             "source": pair[0], "target": pair[1]})
        out_dir = PROCESSED_DIR / "asclepius_bhc"

    parts = split_by_group(rows, "group")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} cặp -> {out_dir / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
