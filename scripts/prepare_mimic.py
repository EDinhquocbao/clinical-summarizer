"""Tạo cặp "các mục khác → Brief Hospital Course" từ MIMIC-IV-Note (discharge.csv.gz).

QUY ĐỊNH DUA: script này KHÔNG in bất kỳ văn bản ghi chú nào, chỉ in số liệu tổng hợp
(số lượng, tỉ lệ, phân vị độ dài, tên các tiêu đề mục phổ biến). Đừng thêm lệnh in nội dung.

Đầu vào : data/raw/mimic-iv-note/discharge.csv.gz
          (cột: note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text)
Đầu ra  : data/processed/mimic_bhc/{train,val,test}.jsonl  ({"id", "group", "source", "target"})
          outputs/results/mimic_stats.json                  (thống kê tổng hợp)
Chia tập: theo subject_id (bệnh nhân), 80/10/10, băm cố định nên chạy lại cho kết quả như cũ.

Chạy:
  python scripts/prepare_mimic.py                      # toàn bộ
  python scripts/prepare_mimic.py --max_notes 20000    # chạy thử nhanh
  python scripts/prepare_mimic.py --section_max_words 150
  python scripts/prepare_mimic.py --max_target_words 250   # chỉ giữ BHC <= 250 từ (vừa sức LSTM)
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from clinsum.corpus import MIMIC_HEADERS, MIMIC_SOURCE_SECTIONS, make_mimic_pair, subject_split
from clinsum.paths import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from clinsum.text import tokenize

# Dòng có dạng tiêu đề ("ADMISSION LABS:") để thống kê tiêu đề con. Chỉ báo cáo tiêu đề xuất hiện
# trong >= 0,5% ghi chú: đó là tiêu đề của mẫu bệnh án, không phải nội dung riêng của bệnh nhân.
GENERIC_HEADER = re.compile(r"^[ \t]*([A-Za-z][A-Za-z /&()\-]{2,40}):[ \t]*$", re.MULTILINE)


def quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    if not xs:
        return {}
    s = sorted(xs)
    out = {f"p{int(q * 100)}": s[min(int(q * len(s)), len(s) - 1)] for q in qs}
    out["mean"] = round(sum(s) / len(s), 1)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RAW_DIR / "mimic-iv-note" / "discharge.csv.gz"))
    ap.add_argument("--out_dir", default=str(PROCESSED_DIR / "mimic_bhc"))
    ap.add_argument("--stats", default=str(OUTPUTS_DIR / "results" / "mimic_stats.json"))
    ap.add_argument("--max_notes", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--min_target_words", type=int, default=30)
    ap.add_argument("--max_target_words", type=int, default=0,
                    help="bỏ ghi chú có BHC dài hơn N từ (0 = không giới hạn); BHC trung vị ~300 từ")
    ap.add_argument("--min_source_words", type=int, default=50)
    ap.add_argument("--section_max_words", type=int, default=0, help="cắt mỗi mục đầu vào tối đa N từ (0 = không cắt)")
    ap.add_argument("--sections", nargs="*", default=list(MIMIC_SOURCE_SECTIONS))
    ap.add_argument("--token_sample", type=int, default=20000, help="số cặp dùng để đo độ dài theo token")
    args = ap.parse_args()

    unknown = [s for s in args.sections if s not in MIMIC_HEADERS]
    if unknown:
        sys.exit(f"Mục không có trong MIMIC_HEADERS: {unknown}")
    src_path = Path(args.input)
    if not src_path.exists() and (RAW_DIR / "discharge.csv.gz").exists():
        src_path = RAW_DIR / "discharge.csv.gz"     # tải thẳng vào data/raw/
    if not src_path.exists():
        sys.exit(f"Không thấy {src_path}. Tải discharge.csv.gz từ PhysioNet vào đó trước.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}

    n_notes = n_pairs = 0
    reasons = Counter()
    present = Counter()                     # số ghi chú có từng mục cấp cao
    sub_headers = Counter()                 # tiêu đề dạng "XXX:" một mình một dòng
    sec_words = {s: [] for s in list(args.sections) + ["brief hospital course"]}
    src_words, tgt_words, src_tok, tgt_tok = [], [], [], []
    split_n, split_subj = Counter(), {"train": set(), "val": set(), "test": set()}
    start = time.time()

    reader = pd.read_csv(src_path, usecols=["note_id", "subject_id", "text"], chunksize=args.chunksize,
                         dtype={"note_id": str, "subject_id": str, "text": str})
    for chunk in reader:
        for note_id, subject_id, text in chunk.itertuples(index=False):
            if args.max_notes and n_notes >= args.max_notes:
                break
            n_notes += 1
            text = text if isinstance(text, str) else ""
            sub_headers.update({h.strip().lower() for h in GENERIC_HEADER.findall(text)})
            pair, sections = make_mimic_pair(text, args.sections, args.min_target_words,
                                             args.min_source_words, args.section_max_words)
            present.update(k for k, v in sections.items() if v)
            for s in sec_words:
                if sections.get(s):
                    sec_words[s].append(len(sections[s].split()))
            if pair is None:
                bhc = sections.get("brief hospital course", "")
                reasons["không có mục Brief Hospital Course" if not bhc else
                        "BHC quá ngắn" if len(bhc.split()) < args.min_target_words else "đầu vào quá ngắn"] += 1
                continue
            source, target = pair
            if args.max_target_words and len(target.split()) > args.max_target_words:
                reasons["BHC quá dài"] += 1
                continue
            split = subject_split(subject_id)
            writers[split].write(json.dumps({"id": note_id, "group": subject_id, "source": source,
                                             "target": target}, ensure_ascii=False) + "\n")
            n_pairs += 1
            split_n[split] += 1
            split_subj[split].add(subject_id)
            src_words.append(len(source.split()))
            tgt_words.append(len(target.split()))
            if len(src_tok) < args.token_sample:
                src_tok.append(len(tokenize(source)))
                tgt_tok.append(len(tokenize(target)))
        else:
            print(f"  đã đọc {n_notes:,} ghi chú, {n_pairs:,} cặp ({time.time() - start:.0f}s)", flush=True)
            continue
        break
    for w in writers.values():
        w.close()

    min_count = max(1, int(0.005 * n_notes))
    stats = {
        "input": src_path.name,
        "notes": n_notes,
        "pairs": n_pairs,
        "pair_rate": round(n_pairs / max(n_notes, 1), 4),
        "dropped": dict(reasons),
        "split_pairs": dict(split_n),
        "split_subjects": {k: len(v) for k, v in split_subj.items()},
        "source_sections": list(args.sections),
        "section_max_words": args.section_max_words,
        "max_target_words": args.max_target_words,
        "section_presence_rate": {k: round(present[k] / max(n_notes, 1), 4) for k in MIMIC_HEADERS},
        "section_words": {k: quantiles(v) for k, v in sec_words.items()},
        "source_words": quantiles(src_words),
        "target_words": quantiles(tgt_words),
        "source_tokens_sample": quantiles(src_tok),
        "target_tokens_sample": quantiles(tgt_tok),
        "compression_ratio_median": round(
            sorted(t / s for s, t in zip(src_words, tgt_words))[len(src_words) // 2], 3) if src_words else None,
        "common_subheaders": {h: c for h, c in sub_headers.most_common(60) if c >= min_count},
        "seconds": round(time.time() - start, 1),
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGhi chú: {n_notes:,} | cặp hợp lệ: {n_pairs:,} ({stats['pair_rate']:.1%})")
    print("Bỏ qua:", dict(reasons))
    print("Chia tập (cặp):", dict(split_n), "| bệnh nhân:", stats["split_subjects"])
    print("Độ dài đầu vào (từ):", stats["source_words"])
    print("Độ dài BHC (từ):     ", stats["target_words"])
    print("Đầu vào (token, mẫu):", stats["source_tokens_sample"])
    print("BHC (token, mẫu):    ", stats["target_tokens_sample"])
    print("Tỉ lệ nén trung vị:", stats["compression_ratio_median"])
    print("Tỉ lệ có từng mục:", {k: v for k, v in stats["section_presence_rate"].items() if v > 0})
    print("Đã ghi", out_dir, "và", args.stats)


if __name__ == "__main__":
    main()
