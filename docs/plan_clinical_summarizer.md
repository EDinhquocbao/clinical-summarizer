# Kế hoạch đồ án cuối kỳ Deep Learning

## Dự án 29: Tóm tắt tự động hồ sơ bệnh án điện tử (Clinical NLP)

*Cập nhật 26/10/2026: chốt phương án A (Pointer-Generator); chạy thử trên dữ liệu tổng hợp Asclepius trong lúc chờ PhysioNet.*

---

## 1. Bài toán và phương pháp

**Input → thuật toán → output**

| | Nội dung |
|---|---|
| Input | Discharge summary (MIMIC: các mục HPI, khám, xét nghiệm, thủ thuật… bỏ mục Brief Hospital Course) |
| Thuật toán | **Pointer-Generator + coverage** (See et al., ACL 2017) |
| Output | Bản tóm tắt diễn tiến điều trị (đích: mục Brief Hospital Course do bác sĩ viết) |

**Vì sao Pointer-Generator:**
- Ghi chú lâm sàng đầy từ hiếm (tên thuốc, liều, mã xét nghiệm) → seq2seq thường sinh `<unk>` hoặc đoán sai; cơ chế **copy** chép thẳng từ văn bản gốc.
- **Coverage** giảm lặp câu, lỗi điển hình của seq2seq.
- Xác suất p_gen cho biết mỗi từ được sinh mới hay chép lại → giải thích được, liên quan trực tiếp tới hallucination.
- Huấn luyện từ đầu, ~14 triệu tham số, chạy được trên RTX 3050 6GB.

**Đối sánh (mục 4.4 theo yêu cầu thầy):**
1. Lead-3 (trích 3 câu đầu) và Oracle trích chọn (cận trên của mọi phương pháp trích chọn).
2. Seq2seq BiLSTM + attention, không copy (ablation: bỏ copy).
3. Pointer-Generator không coverage (ablation: bỏ coverage).
4. (Nếu thầy chấp nhận mô hình tiền huấn luyện) BART-base fine-tune làm mốc tham chiếu trên.
5. Kết quả đã công bố trên bài toán tóm tắt hospital course của MIMIC (so sánh gián tiếp; trích dẫn cần kiểm tra lại).

**Công cụ (không phải phương pháp):** NER (BioBERT/ClinicalBERT fine-tune trên BC5CDR hoặc scispaCy) dùng để trích thực thể khi tính entity recall / hallucination rate.

---

## 2. Dữ liệu

### Dữ liệu tạm (đang dùng): Asclepius
- Ghi chú xuất viện **tổng hợp** do GPT-3.5 sinh từ ca bệnh PMC-Patients (Kweon et al., 2023), công khai, CC-BY-NC-SA 4.0, không có dữ liệu bệnh nhân thật.
- Dùng 19.756 dòng task "Summarization": input = yêu cầu tóm tắt + ghi chú (~350 token), đích = bản tóm tắt (~87 token), tỉ lệ nén ~0,29.
- Chia 80/10/10: 15.804 / 1.975 / 1.977 (mỗi ca bệnh một ghi chú, chia theo ca bệnh).
- Hạn chế: tóm tắt tham chiếu do LLM viết, khá trích chọn (Oracle ROUGE-L ~58); ghi chú ngắn hơn MIMIC nhiều.
- Đã thử cách "các mục khác → Hospital Course" trên Asclepius: mục Hospital Course dài hơn phần còn lại (trung vị 122 so với 107 từ) nên không phải bài toán tóm tắt; cách này để dành cho MIMIC.

### Dữ liệu chính (chờ quyền): MIMIC-IV-Note
- Cặp (các mục khác của discharge summary → Brief Hospital Course), chia theo bệnh nhân (`subject_id`).
- Code dùng chung: `clinsum.corpus.make_pair` (đã có test), chỉ cần đổi nguồn văn bản.
- MIMIC-III: kiểm tra bên ngoài nếu ký thêm DUA.
- Nên xem thêm các bộ dẫn xuất đã ghép sẵn cặp input/BHC trên PhysioNet (cần kiểm tra tên và điều kiện truy cập).

### n2c2 / i2b2: tạm không khả dụng
- Cổng DBMI báo "temporarily unavailable", từng lỗi 500; đã gửi email qua Contact Us, chưa có phản hồi. Không còn bắt buộc với phương án A.

---

## 3. Tiến độ

| Hạng mục | Trạng thái |
|---|---|
| PhysioNet credentialing | ✅ Duyệt 18/10 |
| CITI (2 khóa) | ✅ Xong, nộp báo cáo 19/10 |
| PhysioNet duyệt báo cáo CITI | ⏳ Chờ từ 19/10 |
| Ký DUA MIMIC-IV-Note / MIMIC-III | ☐ Sau khi duyệt CITI |
| Repo, môi trường, torch CUDA (RTX 3050 6GB) | ✅ |
| Tiền xử lý (placeholder, tách mục, cặp BHC, chia theo bệnh nhân) | ✅ Có test |
| Dữ liệu tạm Asclepius | ✅ |
| Baseline Lead-3, Oracle | ✅ |
| Seq2seq-attn, Pointer-Generator, + coverage | ✅ Cài đặt và chạy thử (kết quả: `ket_qua_so_bo.md`) |
| Độ đo: ROUGE, novel n-gram, lặp từ, recall/hallucination số liệu | ✅ |
| NER làm công cụ đánh giá thực thể | ☐ |
| Chạy trên MIMIC-IV-Note | ☐ Chờ dữ liệu |
| Phân tích lỗi, đánh giá thủ công 20–30 mẫu | ☐ |
| Demo web (văn bản tự soạn) | ☐ |
| Báo cáo cuối kỳ | ☐ |

---

## 4. Việc tiếp theo

1. Khi có MIMIC: trích discharge summary, `make_pair`, EDA độ dài, chỉnh `max_source_len` (MIMIC dài hơn nhiều: chọn mục đầu vào hoặc cắt), huấn luyện lại 3 mô hình.
2. NER làm công cụ: fine-tune BioBERT trên BC5CDR (bệnh, hóa chất/thuốc) → entity recall / hallucination rate cho thuốc và bệnh.
3. Tinh chỉnh: độ dài đầu vào, trọng số coverage, beam size, chặn lặp trigram.
4. Hỏi thầy: có chấp nhận BART-base làm mốc tham chiếu trên không.

---

## 5. Rủi ro

| Rủi ro | Cách giảm thiểu |
|---|---|
| PhysioNet duyệt chậm | Toàn bộ pipeline đã chạy trên Asclepius; chỉ thay dữ liệu |
| Văn bản MIMIC quá dài cho LSTM | Chọn mục đầu vào (HPI, kết quả, thủ thuật), cắt 400–800 token |
| ROUGE-L ≥ 0.45 khó đạt trên BHC thật | Nêu rõ định nghĩa đích; báo cáo thêm entity recall, hallucination |
| Dữ liệu tạm do LLM sinh | Nói rõ là chạy thử pipeline, không kết luận lâm sàng |
| Vi phạm quy định dữ liệu | Không dán văn bản MIMIC vào LLM/API; chỉ chia sẻ số liệu tổng hợp |
