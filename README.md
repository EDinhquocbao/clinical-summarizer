# Clinical Summarizer: Tóm tắt tự động hồ sơ bệnh án điện tử

Đồ án cuối kỳ môn Deep Learning, Dự án 29 (Clinical NLP, lĩnh vực Y tế).

Hệ thống kết hợp **trích xuất thông tin y tế** (NER, Relation Extraction) và **tóm tắt trừu tượng**
để tóm tắt hồ sơ bệnh án dài, tạo báo cáo xuất viện có cấu trúc, đồng thời dùng NER để kiểm tra
chéo và phát hiện thông tin bị mô hình "bịa" (hallucination).

## Mục tiêu
- ROUGE-L >= 0.45 (chỉ tiêu đề gốc; sẽ nêu rõ định nghĩa "tóm tắt chuẩn")
- Trích xuất đúng >= 90% thực thể y tế quan trọng (báo cáo F1 theo từng loại)
- Demo ứng dụng web

## Mô hình
BioBERT, ClinicalBERT (NER/RE); BART, T5, LED (tóm tắt).

## Dữ liệu
MIMIC-III / MIMIC-IV-Note, MIMIC-IV-Ext-EntityCoding (PhysioNet, cần credentialing + DUA);
BC5CDR, NCBI-Disease (công khai). n2c2/i2b2 hiện tạm không khả dụng.
Chi tiết: `docs/plan_clinical_summarizer.md`.

> **Quy định dữ liệu:** không đưa dữ liệu lâm sàng, kết quả sinh từ dữ liệu thật hay văn bản bệnh án thật
> lên repo, slide hoặc demo. Không dán dữ liệu vào dịch vụ mô hình ngôn ngữ bên thứ ba.

## Cấu trúc
```
clinical-summarizer/
├── configs/        # cấu hình huấn luyện (YAML)
├── data/           # raw/ interim/ processed/ external/  (KHÔNG commit)
├── docs/           # kế hoạch, ghi chú semina
├── models/         # checkpoint (KHÔNG commit)
├── notebooks/      # EDA, thử nghiệm
├── outputs/        # hình, kết quả
├── scripts/        # check_env.py, download_open_data.py
├── src/clinsum/    # mã nguồn chính (preprocess, ner/, summarization/, evaluate, pipeline)
├── tests/          # kiểm thử
└── app/            # web demo
```

## Bắt đầu nhanh (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
# Cài PyTorch theo máy: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
pytest
```

## Tiến độ
Xem `docs/plan_clinical_summarizer.md` (mục 9).
