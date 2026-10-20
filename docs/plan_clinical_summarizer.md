# Kế hoạch đồ án cuối kỳ Deep Learning

## Dự án 29: Tóm tắt tự động hồ sơ bệnh án điện tử (Clinical NLP)

*Phiên bản cập nhật ngày 18/10/2026: n2c2/i2b2 tạm không khả dụng, chuyển sang bộ dữ liệu thay thế trên PhysioNet.*

---

## 1. Tổng quan đề tài

**Mô tả:** Xây dựng hệ thống NLP tự động tóm tắt hồ sơ bệnh án dài, trích xuất thông tin y tế quan trọng (chẩn đoán, thuốc, kết quả xét nghiệm) và tạo báo cáo xuất viện có cấu trúc.

**Phương pháp:** BART, T5, ClinicalBERT, BioBERT, NER y tế, Relation Extraction, Abstractive Summarization.

**Chỉ tiêu kết quả (đề gốc):**
- ROUGE-L ≥ 0.45
- Trích xuất đúng ≥ 90% thực thể y tế quan trọng
- Demo ứng dụng web tóm tắt hồ sơ

### Hai bài toán con

1. **Information Extraction (IE):** NER (bệnh, thuốc, xét nghiệm, liều dùng) và Relation Extraction.
2. **Abstractive Summarization:** sinh bản tóm tắt dạng báo cáo xuất viện từ hồ sơ dài.

Ý tưởng chính: dùng kết quả NER để **dẫn hướng** mô hình tóm tắt, giảm bỏ sót và giảm hallucination.

---

## 2. Tình hình dữ liệu (cập nhật)

### n2c2 / i2b2: tạm không khả dụng
- Trang n2c2 NLP trên DBMI Data Portal ghi dữ liệu "temporarily unavailable"; đăng ký không mở; trang dự án từng báo lỗi Server Error (500).
- Các track 2022 đã đóng đăng ký. Mayo Clinic chỉ phát hành 2019 Track 1 (Clinical STS) và Track 2 (Family History), không phù hợp đề tài.
- Không rõ lý do và thời điểm mở lại. Đã gửi email hỏi qua form Contact Us (chờ phản hồi).
- **Quyết định:** n2c2 chỉ là nguồn bổ trợ, không bắt buộc. Nếu mở lại sẽ xin n2c2 2018 Track 2 và i2b2 2010.
- Lưu ý: n2c2 2018 Track 2 gồm 505 discharge summary lấy từ MIMIC-III, 9 loại thực thể (ADE, Dosage, Duration, Drug, Form, Frequency, Reason, Route, Strength). Dùng bài tổng quan Track 2 (Henry et al., JAMIA 2020, PMC7489085) làm tài liệu tham khảo và mốc so sánh (F1 tốt nhất khoảng 0.94 cho trích xuất khái niệm).

### Bộ dữ liệu thay thế

| Mục đích | Bộ dữ liệu | Nguồn | Ghi chú |
|---|---|---|---|
| Văn bản tóm tắt | MIMIC-IV-Note (discharge) hoặc MIMIC-III NOTEEVENTS | PhysioNet | Xem mục "Quyết định cần chốt" |
| NER + assertion | MIMIC-IV-Ext-EntityCoding (400 discharge summary, hơn 85 nghìn thực thể) | PhysioNet | Gần i2b2 2010 nhất; nhãn thiên về mã hóa ICD, cần xem danh sách loại thực thể |
| NER (thuốc/hóa chất, bệnh) | BC5CDR, NCBI-Disease | Công khai | Nhãn chuẩn, nhưng là văn bản y văn |
| NER mở rộng (tùy chọn) | SNOMED CT Entity Linking Challenge (272 discharge summary) | PhysioNet | NER + liên kết SNOMED, khó hơn |
| NER + RE (phụ, tùy chọn) | NIHSS Annotations for MIMIC-III | PhysioNet | Chủ đề hẹp (đột quỵ) |
| Đánh giá hallucination | MIMIC-III-Ext-VeriFact-BHC | PhysioNet | Mới xem mô tả; cần kiểm tra cách dùng |
| Nhãn bổ sung thuốc/liều | Gán nhãn tự động (scispaCy/MedCAT) trên MIMIC | Tự tạo | Nhãn "bạc", cần kiểm tra thủ công mẫu nhỏ |

**Cần kiểm tra mục Files của từng bộ:** điều kiện truy cập, DUA riêng (các bộ dựa trên MIMIC-IV-Note có thể cần ký thêm).

### Quyết định cần chốt
- [ ] Dùng **MIMIC-IV-Note** cho cả NER và tóm tắt (để cùng nguồn văn bản) hay giữ MIMIC-III theo đề gốc? Thầy đã đồng ý hướng đổi, cần xác nhận lựa chọn cụ thể.

---

## 3. Kiến trúc pipeline đề xuất

```
Hồ sơ bệnh án (raw text)
        │
        ▼
[1] Tiền xử lý: xử lý placeholder de-id, tách section
        │
        ▼
[2] NER / Relation Extraction (BioBERT / ClinicalBERT)
        │
        ▼
[3] Tóm tắt có điều kiện (BART / T5 / LED + thực thể quan trọng)
        │
        ▼
[4] Hậu xử lý + kiểm tra chéo bằng NER (phát hiện hallucination)
        │
        ▼
[5] Báo cáo xuất viện có cấu trúc
        │
        ▼
[6] Web demo (Gradio / Streamlit)
```

---

## 4. Tổng quan các giai đoạn

| GĐ | Nội dung | Kết quả đầu ra | Trạng thái |
|----|----------|----------------|-----------|
| 0 | Xin quyền dữ liệu | Truy cập MIMIC + bộ PhysioNet | Đang làm |
| 1 | Môi trường và cấu trúc dự án | Repo chạy được, có dữ liệu mở để thử | Chưa bắt đầu |
| 2 | Tiền xử lý và EDA | Phân bố độ dài, các section, tập train/val/test | Chờ dữ liệu |
| 3 | Baseline (Lead-N, BART-base chưa fine-tune) | Điểm ROUGE mốc | Chưa bắt đầu |
| 4 | Mô hình NER (BioBERT/ClinicalBERT) | F1 theo từng loại thực thể | Chưa bắt đầu |
| 5 | Mô hình tóm tắt: (a) thuần, (b) có hướng dẫn bằng thực thể | Bảng so sánh ablation | Chưa bắt đầu |
| 6 | Đánh giá và phân tích lỗi | Phần kết quả của báo cáo | Chưa bắt đầu |
| 7 | Web demo và báo cáo | Sản phẩm cuối kỳ | Chưa bắt đầu |

---

## 5. Chi tiết từng giai đoạn

### GĐ 0: Xin quyền dữ liệu

**PhysioNet (credentialing)**
- [x] Tạo tài khoản PhysioNet
- [x] Nộp hồ sơ credentialing (người tham chiếu: trưởng ngành Khoa học máy tính)
- [x] **Hồ sơ credentialing đã được duyệt** (email ngày 18/10/2026)
- [x] Khóa CITI "Conflicts of Interest" hoàn thành
- [ ] Khóa CITI "Data or Specimens Only Research" (đang 6/9 module)
- [ ] Tải Completion Report (Records → View/Print), **không phải certificate**
- [ ] Nộp báo cáo lên PhysioNet (Settings → Training), chờ duyệt
- [ ] Ký DUA: MIMIC-III và/hoặc MIMIC-IV-Note
- [ ] Ký DUA / xin truy cập: MIMIC-IV-Ext-EntityCoding (và các bộ khác nếu dùng)
- [ ] Tải dữ liệu vào `data/raw/` (không đưa lên git)

**Quy tắc PhysioNet cần nhớ (từ email duyệt)**
- Chỉ người được cấp quyền mới truy cập; không chia sẻ dữ liệu.
- Nếu làm nhóm, mỗi người tự làm credentialing.
- Việc dùng LLM với dữ liệu credentialed phải tuân thủ DUA và hướng dẫn tại physionet.org/news/post/llm-responsible-use/. **Không dán văn bản bệnh án thật vào chatbot hoặc API bên thứ ba.**

**n2c2 (tùy chọn, chờ mở lại)**
- [x] Tạo tài khoản DBMI Data Portal
- [x] Gửi email hỏi qua Contact Us
- [ ] Kiểm tra lại trang n2c2 định kỳ; nếu mở thì xin 2018 Track 2 và i2b2 2010

### GĐ 1: Môi trường và cấu trúc dự án

```
clinical-summarizer/
├── data/            # raw/, processed/ (KHÔNG đưa lên git)
├── notebooks/       # EDA, thử nghiệm
├── src/
│   ├── preprocess.py
│   ├── ner_train.py
│   ├── sum_train.py
│   ├── evaluate.py
│   └── pipeline.py
├── app/             # web demo
├── configs/
└── requirements.txt
```

**Thư viện:** `transformers`, `datasets`, `torch`, `evaluate`, `rouge-score`, `bert-score`, `seqeval`, `peft`, `gradio`

- [ ] Tạo repo, `.gitignore` (loại trừ `data/`)
- [ ] Viết `requirements.txt`, cài môi trường
- [ ] Script tải BC5CDR / NCBI-Disease
- [ ] Chạy thử fine-tune NER nhỏ để kiểm tra môi trường

### GĐ 2: Tiền xử lý và EDA

- [ ] Đọc và làm sạch discharge summary
- [ ] Xử lý placeholder ẩn danh (dạng `___` hoặc `[**Name**]` tùy phiên bản MIMIC)
- [ ] Tách section: History of Present Illness, Hospital Course, Medications, Discharge Diagnosis...
- [ ] Định nghĩa "tóm tắt chuẩn" (ví dụ mục Brief Hospital Course làm đích), ghi rõ trong báo cáo
- [ ] EDA: phân bố độ dài (token), tỉ lệ nén, các section phổ biến
- [ ] Chia train/val/test **theo bệnh nhân** (tránh rò rỉ dữ liệu)

### GĐ 3: Baseline

- [ ] Lead-N / Lead-3 (extractive)
- [ ] BART-base chưa fine-tune (zero-shot)
- [ ] Viết `evaluate.py`: ROUGE-1/2/L, BERTScore
- [ ] Ghi lại điểm mốc

### GĐ 4: Mô hình NER và Relation Extraction (cập nhật)

**Chiến lược 3 lớp (thay cho n2c2):**
1. **Huấn luyện/đánh giá có nhãn chuẩn:** BC5CDR + NCBI-Disease (bệnh, hóa chất/thuốc) và MIMIC-IV-Ext-EntityCoding (NER + assertion trên discharge summary).
2. **Nhãn tự động trên MIMIC** (scispaCy/MedCAT) cho các loại thuốc/liều/xét nghiệm mà bộ có nhãn chuẩn không bao phủ.
3. **Kiểm tra thủ công** khoảng 50 đến 100 câu để ước lượng độ chính xác trên văn bản lâm sàng thật; báo cáo rõ đây là nhãn tự động.

- [ ] Fine-tune BioBERT và ClinicalBERT (token classification)
- [ ] Đánh giá bằng seqeval: Precision / Recall / F1 **theo từng loại thực thể**
- [ ] So sánh BioBERT và ClinicalBERT
- [ ] (Mở rộng) Assertion classification (có/không có); Relation Extraction (thuốc ↔ liều)
- [ ] Đối chiếu với mốc n2c2 2018 Track 2 trong tài liệu (F1 tốt nhất khoảng 0.94 cho trích xuất khái niệm; ADE và Reason khó hơn)

### GĐ 5: Mô hình tóm tắt

**Xử lý văn bản dài** (BART giới hạn 1024 token), chọn một hoặc kết hợp:
- Chunking rồi tóm tắt phân cấp
- LED (Longformer-Encoder-Decoder)
- Tóm tắt riêng theo từng section

| Thí nghiệm | Mô tả |
|-----------|-------|
| (a) Tóm tắt thuần | BART/T5/LED fine-tune trên văn bản gốc |
| (b) Có hướng dẫn bằng thực thể | Đưa danh sách thực thể quan trọng vào đầu vào |
| (c) (Tùy chọn) Có hậu kiểm | Dùng NER kiểm tra chéo, loại thực thể không có trong bản gốc |

- [ ] Fine-tune (mixed precision, LoRA nếu thiếu VRAM)
- [ ] Tinh chỉnh siêu tham số cơ bản
- [ ] Bảng kết quả so sánh

### GĐ 6: Đánh giá và phân tích lỗi

- [ ] ROUGE-1/2/L, BERTScore
- [ ] **Entity recall:** thực thể quan trọng của bản gốc còn trong bản tóm tắt
- [ ] **Hallucination rate:** thuốc/liều/chẩn đoán xuất hiện trong tóm tắt nhưng không có trong bản gốc
- [ ] Tham khảo cách đánh giá của MIMIC-III-Ext-VeriFact-BHC (kiểm chứng mệnh đề của tóm tắt Brief Hospital Course)
- [ ] Đánh giá thủ công trên mẫu nhỏ (20 đến 30 mẫu)
- [ ] Phân tích lỗi bằng ví dụ cụ thể
- [ ] Đối chiếu chỉ tiêu: ROUGE-L ≥ 0.45, trích xuất ≥ 90% (diễn giải theo từng loại thực thể và bộ dữ liệu thực tế)

### GĐ 7: Web demo và báo cáo

- [ ] Demo Gradio/Streamlit: nhập hồ sơ → thực thể được đánh dấu, tóm tắt, báo cáo có cấu trúc
- [ ] Demo chỉ dùng văn bản mẫu tự tạo, **không dùng bản ghi thật từ MIMIC**
- [ ] Viết báo cáo, làm slide
- [ ] Quay video demo dự phòng

---

## 6. Cấu trúc báo cáo gợi ý

1. Giới thiệu và động lực
2. Các nghiên cứu liên quan (ClinicalBERT, BioBERT, BART, T5, LED, n2c2 2018 Track 2)
3. Dữ liệu và tiền xử lý (ghi rõ n2c2 không khả dụng và cách thay thế)
4. Phương pháp (NER/RE, tóm tắt, pipeline)
5. Thiết lập thí nghiệm
6. Kết quả và ablation
7. Phân tích lỗi, hallucination, hạn chế
8. Đạo đức và giới hạn sử dụng
9. Kết luận và hướng phát triển

---

## 7. Rủi ro và cách giảm thiểu

| Rủi ro | Cách giảm thiểu |
|--------|-----------------|
| n2c2 không mở lại | Dùng EntityCoding + BC5CDR + nhãn tự động; báo cáo minh bạch |
| Mất mốc so sánh chuẩn như n2c2 | Trích mốc từ bài báo n2c2 2018 Track 2; so sánh giữa BioBERT và ClinicalBERT trên cùng tập |
| Nhãn tự động có nhiễu | Kiểm tra thủ công mẫu nhỏ, báo cáo sai số |
| Duyệt dữ liệu/DUA chậm | Làm trước trên dữ liệu mở; nộp sớm |
| Văn bản quá dài | Chunking, LED, tách section |
| Hallucination | Hướng dẫn bằng thực thể, hậu kiểm bằng NER |
| ROUGE-L ≥ 0.45 khó đạt | Nêu rõ định nghĩa tóm tắt chuẩn; báo cáo thêm BERTScore, entity recall |
| Thiếu GPU | Colab/Kaggle, bản base, mixed precision, LoRA |
| Vi phạm quy định dữ liệu | Không chia sẻ, không dán văn bản thật vào LLM/API bên thứ ba, không đưa mẫu thật vào demo |

---

## 8. Thông tin cần bổ sung

- [ ] GPU đang có (hoặc dùng Colab/Kaggle)
- [ ] Số tuần còn lại đến hạn nộp
- [ ] Làm cá nhân hay nhóm (nếu nhóm, mỗi người tự credentialing)
- [ ] Chốt MIMIC-III hay MIMIC-IV-Note

---

## 9. Tiến độ

| Hạng mục | Trạng thái |
|---|---|
| PhysioNet credentialing | ✅ Đã duyệt |
| CITI: Conflicts of Interest | ✅ Xong |
| CITI: Data or Specimens Only Research | ⏳ 6/9 module |
| Nộp Completion Report lên PhysioNet | ☐ |
| Ký DUA MIMIC (III / IV-Note) | ☐ |
| Truy cập MIMIC-IV-Ext-EntityCoding | ☐ |
| n2c2 / i2b2 | ⏸ Tạm đóng, đã gửi email hỏi |
| GĐ 1 trở đi | ☐ Chưa bắt đầu |
