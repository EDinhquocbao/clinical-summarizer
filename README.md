# Clinical Summarizer: Tóm tắt tự động hồ sơ bệnh án điện tử

Đồ án cuối kỳ môn Deep Learning, Dự án 29 (Clinical NLP, lĩnh vực Y tế).

**Bài toán:** discharge summary (báo cáo xuất viện) dài → bản tóm tắt ngắn, giữ đúng chẩn đoán, điều trị, thuốc, liều.

**Phương pháp chính:** Pointer-Generator + coverage (See, Liu, Manning, ACL 2017): seq2seq BiLSTM có
attention, thêm cơ chế **copy** (chép nguyên từ hiếm như tên thuốc, liều từ văn bản gốc) và **coverage**
(chống lặp ý). Huấn luyện từ đầu, vừa với GPU 6GB.

**Đối sánh:** Lead-3 (trích 3 câu đầu), Oracle trích chọn (cận trên), Seq2seq + attention không copy.
NER/BioBERT/ClinicalBERT/BART là **công cụ** (trích thực thể để đánh giá), không phải phương pháp chính.

## Dữ liệu
| Bộ | Vai trò | Trạng thái |
|---|---|---|
| Asclepius synthetic notes (HF `starmpcc/Asclepius-Synthetic-Clinical-Notes`, CC-BY-NC-SA 4.0) | Dữ liệu **tạm**: ghi chú xuất viện tổng hợp → tóm tắt (19.756 cặp) | Đang dùng |
| MIMIC-IV-Note (discharge) | Dữ liệu **chính**: các mục khác → Brief Hospital Course | Chờ PhysioNet duyệt CITI, ký DUA |
| MIMIC-III NOTEEVENTS | Kiểm tra bên ngoài (tùy chọn) | Chờ DUA |
| BC5CDR | Huấn luyện NER làm công cụ đánh giá | Tải được |

> **Quy định dữ liệu:** không đưa dữ liệu lâm sàng, kết quả sinh từ dữ liệu thật hay văn bản bệnh án thật
> lên repo, slide hoặc demo. Không dán dữ liệu MIMIC vào dịch vụ mô hình ngôn ngữ bên thứ ba.

## Cấu trúc
```
clinical-summarizer/
├── configs/        # summarization_base + seq2seq_attn / pointer_gen / pointer_gen_cov, ner
├── data/           # raw/ interim/ processed/ external/  (KHÔNG commit)
├── docs/           # kế hoạch, cấu trúc trình bày, ghi chú semina
├── models/         # checkpoint, từ điển (KHÔNG commit)
├── outputs/        # logs/, predictions/, results/, figures/
├── scripts/        # check_env, download_open_data, prepare_asclepius, score_predictions, plot_training
├── src/clinsum/    # preprocess, corpus, text, evaluate, summarization/ (model, data, vocab, train, predict, baselines)
└── tests/
```

## Chạy (Windows, từ thư mục gốc repo)
```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File C:\Users\dinhq\wheels\install-torch.ps1   # torch bản CUDA (cu126)
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
pytest

# 1. Dữ liệu tạm (tải synthetic.csv vào data/external/asclepius/ trước)
python scripts/prepare_asclepius.py
# 2. Baseline
python -m clinsum.summarization.baselines --limit 1000
# 3. Huấn luyện
python -m clinsum.summarization.train --config configs/seq2seq_attn.yaml --set max_minutes=45
python -m clinsum.summarization.train --config configs/pointer_gen.yaml --set max_minutes=45
python -m clinsum.summarization.train --config configs/pointer_gen_cov.yaml --set max_minutes=10
# 4. Sinh tóm tắt và chấm điểm
python -m clinsum.summarization.predict --checkpoint models/pointer_gen_cov/best.pt --limit 1000
python scripts/score_predictions.py
```

## Kết quả và tiến độ
Kết quả sơ bộ: `docs/ket_qua_so_bo.md`. Kế hoạch và tiến độ: `docs/plan_clinical_summarizer.md`.
