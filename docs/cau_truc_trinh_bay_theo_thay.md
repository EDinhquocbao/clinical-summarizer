# Cấu trúc trình bày theo yêu cầu của thầy

*Ghi lại từ lúc thầy nhận xét bài seminar của một bạn khác. Dùng làm khung cho mọi buổi seminar và báo cáo cuối kỳ.*
*Cập nhật 26/10/2026 theo phương án A (Pointer-Generator).*

## Cấu trúc bắt buộc

1. **Giới thiệu bài toán:** input → thuật toán → output
2. **Các nghiên cứu liên quan**
3. **Phương pháp giải quyết:** chọn **1** phương pháp
4. **Thử nghiệm**: 4.1 Dữ liệu (bảng dữ liệu, tiền xử lý) · 4.2 Độ đo · 4.3 Kết quả · 4.4 Đối sánh

Thầy nhấn mạnh: các mục trong bảng đề tài (NER, Relation Extraction, BART, T5, BioBERT, ClinicalBERT…) là **công cụ**;
"phương pháp" là **một mô hình học sâu giải bài toán lõi** (tóm tắt), chọn 1 làm chính, 1–2 làm đối sánh.

---

## Ánh xạ vào đề tài

### 1. Bài toán
| | Nội dung |
|---|---|
| **Input** | Discharge summary dài (bỏ mục cần tóm tắt) |
| **Thuật toán** | Pointer-Generator + coverage |
| **Output** | Đoạn tóm tắt diễn tiến điều trị (Brief Hospital Course) |

Một sơ đồ một dòng + một ví dụ **tự soạn hoặc từ dữ liệu tổng hợp** (không dùng văn bản MIMIC thật).

### 2. Nghiên cứu liên quan (cần kiểm tra lại trích dẫn trước khi đưa lên slide)
- **Tóm tắt seq2seq có attention:** Bahdanau et al. 2015 (attention); Rush et al. 2015; Nallapati et al. 2016.
- **Copy / pointer:** Vinyals et al. 2015 (Pointer Networks); Gu et al. 2016 (CopyNet); **See et al. 2017** (Pointer-Generator + coverage) ← phương pháp chính.
- **Tóm tắt văn bản lâm sàng:** Adams et al. 2021 (hospital course summarization, MIMIC); shared task BioNLP 2024 "Discharge Me!".
- **Mô hình tiền huấn luyện (công cụ / đối sánh tham chiếu):** BART, T5, ClinicalBERT, BioBERT.
- **Dữ liệu:** MIMIC-III (Johnson et al. 2016), MIMIC-IV; Asclepius (Kweon et al. 2023).

### 3. Phương pháp: Pointer-Generator + coverage
- Encoder BiLSTM, decoder LSTM, attention Bahdanau.
- p_gen ∈ (0, 1) trộn hai nguồn: sinh từ từ điển và chép từ đầu vào (qua attention).
- Coverage vector = tổng attention các bước trước; loss coverage phạt chú ý lặp lại một chỗ.
- Lý do chọn: từ hiếm trong y khoa, lặp ý, giải thích được, vừa GPU 6GB.

### 4. Thử nghiệm
- **4.1 Dữ liệu:** bảng (Asclepius tạm thời, MIMIC-IV-Note chính), tiền xử lý (placeholder, tách mục, cắt độ dài, chia theo bệnh nhân).
- **4.2 Độ đo:** ROUGE-1/2/L; novel bigram (mức trừu tượng); lặp trigram; recall / hallucination của số liệu (liều, tuổi, xét nghiệm); sau này entity recall / hallucination của thuốc, bệnh bằng NER.
- **4.3 Kết quả:** bảng các mô hình, đường cong loss, ví dụ đầu ra.
- **4.4 Đối sánh:** Lead-3, Oracle, Seq2seq-attn (bỏ copy), PG (bỏ coverage); kết quả công bố trên MIMIC (gián tiếp).

---

## Câu hỏi nên hỏi thầy
1. Có chấp nhận BART-base fine-tune làm **mốc tham chiếu** (không phải phương pháp chính) không?
2. Kết quả trên dữ liệu tổng hợp (Asclepius) trong lúc chờ MIMIC có được tính là thử nghiệm sơ bộ không?
3. Đối sánh với benchmark công bố (khác bộ dữ liệu) có cần thiết không, hay chỉ đối sánh nội bộ?

## Checklist trước mỗi buổi trình bày
- [ ] Đủ 4 phần đúng thứ tự
- [ ] Phần 1 có sơ đồ input → thuật toán → output
- [ ] Phần 2 nêu điểm khác của đề tài so với từng nghiên cứu
- [ ] Phần 3 chỉ nói **một** phương pháp chính
- [ ] Phần 4.1 có bảng dữ liệu và tiền xử lý
- [ ] Phần 4.4 nêu rõ đối sánh với cái gì
- [ ] Không có văn bản bệnh án thật trên slide
