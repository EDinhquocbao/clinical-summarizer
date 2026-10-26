"""Tải dữ liệu NER công khai từ Hugging Face (dùng làm công cụ trích thực thể khi đánh giá tóm tắt).

`datasets` >= 4 không còn chạy loading script (file .py trong repo dataset), nên ở đây đọc bản
parquet Hugging Face tự chuyển đổi (nhánh refs/convert/parquet).
- BC5CDR (tner/bc5cdr): có bản parquet -> tải được.
- NCBI-Disease (ncbi/ncbi_disease): chỉ có loading script, chưa có bản parquet -> tạm bỏ qua.

Dữ liệu lưu ở data/external/ (không đưa lên Git).
Chạy:  python scripts/download_open_data.py
"""
import sys

from datasets import load_dataset

from clinsum.paths import EXTERNAL_DIR, ensure_dirs

sys.stdout.reconfigure(encoding="utf-8")

CANDIDATES = {
    "bc5cdr": ("tner/bc5cdr", "refs/convert/parquet"),
}
SKIPPED = {"ncbi_disease": "ncbi/ncbi_disease chỉ có loading script, chưa có bản parquet trên Hugging Face"}


def main() -> None:
    ensure_dirs()
    for key, (hf_name, revision) in CANDIDATES.items():
        try:
            ds = load_dataset(hf_name, revision=revision)
            ds.save_to_disk(str(EXTERNAL_DIR / key))
            print(f"[OK] {key}: {ds}")
        except Exception as e:  # noqa: BLE001
            print(f"[LỖI] {key} ({hf_name}): {e}")
    for key, why in SKIPPED.items():
        print(f"[BỎ QUA] {key}: {why}")


if __name__ == "__main__":
    main()
