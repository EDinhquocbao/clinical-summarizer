"""Fine-tune BioBERT/ClinicalBERT cho NER (token classification).

TODO (Giai đoạn 4):
1. Tải dataset theo cấu hình (BC5CDR / NCBI-Disease; sau này EntityCoding).
2. Căn chỉnh nhãn với subword token (word_ids của tokenizer).
3. Huấn luyện bằng transformers.Trainer, đánh giá bằng seqeval theo từng loại thực thể.
Chạy dự kiến:  python -m clinsum.ner.train --config configs/ner.yaml
"""


def main() -> None:
    raise NotImplementedError("Chưa cài đặt: xem TODO ở đầu file.")


if __name__ == "__main__":
    main()
