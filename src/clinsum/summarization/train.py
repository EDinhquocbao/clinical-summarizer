"""Fine-tune BART/T5/LED cho tóm tắt.

TODO (Giai đoạn 3 và 5):
1. Baseline: Lead-3 và mô hình chưa fine-tune.
2. Xử lý văn bản dài: chia đoạn phân cấp, LED, hoặc tóm tắt theo mục.
3. Thí nghiệm (a) tóm tắt thuần, (b) có dẫn hướng thực thể (use_entity_guidance: true).
Chạy dự kiến:  python -m clinsum.summarization.train --config configs/summarization.yaml
"""


def main() -> None:
    raise NotImplementedError("Chưa cài đặt: xem TODO ở đầu file.")


if __name__ == "__main__":
    main()
