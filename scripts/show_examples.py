"""In ví dụ đầu ra để phân tích lỗi: tham chiếu và dự đoán của từng mô hình cho cùng một ghi chú.

Mặc định chọn các mẫu mà tham chiếu chứa số liệu có trong ghi chú (liều, tuổi…) để xem mô hình
nào giữ được. CHỈ dùng với dữ liệu tổng hợp/công khai, không in văn bản MIMIC ra slide.
Chạy:  python scripts/show_examples.py --n 5
"""
import argparse
import sys

from clinsum.paths import OUTPUTS_DIR, ROOT
from clinsum.text import numbers, tokenize
from clinsum.summarization.data import read_jsonl

MODELS = ["lead3", "seq2seq_attn", "pointer_gen", "pointer_gen_cov"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/asclepius_sum")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    rows = {r["id"]: r for r in read_jsonl(ROOT / args.data_dir / "test.jsonl")}
    preds = {}
    for m in MODELS:
        path = OUTPUTS_DIR / "predictions" / f"{m}_test.jsonl"
        if path.exists():
            preds[m] = {p["id"]: p for p in read_jsonl(path)}
    common = [i for i in next(iter(preds.values())) if all(i in p for p in preds.values())]
    if args.ids:
        chosen = args.ids
    else:
        chosen = [i for i in common
                  if set(numbers(tokenize(rows[i]["target"]))) & set(tokenize(rows[i]["source"]))][: args.n]
    for i in chosen:
        r = rows[i]
        print("=" * 100, "\nID", i, "| nguồn", len(tokenize(r["source"])), "token")
        print("[REQUEST]", r["source"].split("\n", 1)[0][:300])
        print("[THAM CHIẾU]", r["target"])
        for m, p in preds.items():
            pg = p[i].get("p_gen")
            print(f"[{m}{'' if pg is None else f' p_gen={pg:.2f}'}]", p[i]["prediction"])


if __name__ == "__main__":
    main()
