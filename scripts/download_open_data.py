"""Tải dữ liệu NER công khai (BC5CDR, NCBI-Disease) từ Hugging Face để chạy thử pipeline.

LƯU Ý: script này CHƯA được chạy thử. Tên dataset trên Hugging Face hoặc cách tải có thể thay đổi
theo phiên bản thư viện `datasets`; nếu báo lỗi, tìm tên dataset hiện hành trên huggingface.co/datasets.
Dữ liệu công khai nên cho phép lưu trong data/external/ (vẫn không đưa lên Git).
Chạy:  python scripts/download_open_data.py
"""
from datasets import load_dataset

from clinsum.paths import EXTERNAL_DIR, ensure_dirs

CANDIDATES = {
    "bc5cdr": "tner/bc5cdr",
    "ncbi_disease": "ncbi/ncbi_disease",
}


def main() -> None:
    ensure_dirs()
    for key, hf_name in CANDIDATES.items():
        try:
            ds = load_dataset(hf_name)
            ds.save_to_disk(str(EXTERNAL_DIR / key))
            print(f"[OK] {key}: {ds}")
        except Exception as e:  # noqa: BLE001
            print(f"[LỖI] {key} ({hf_name}): {e}")


if __name__ == "__main__":
    main()
