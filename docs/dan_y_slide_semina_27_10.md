# Dàn ý slide semina 27/10/2026 (phương án A: Pointer-Generator)

> **TRẠNG THÁI: CHƯA LÀM SLIDE.** Đây mới là dàn ý.
> Chỉ bắt đầu làm slide **sau khi session chính chạy xong** (huấn luyện + đánh giá trên dữ liệu thay thế),
> rồi điền các ô `[kết quả]` bằng số liệu thật. Không bịa số liệu.

Khoảng 12–14 slide, 12–15 phút, theo đúng 4 phần thầy yêu cầu.

## 0. Bìa
Tên đề tài, Dự án 29, tên sinh viên, ngày 27/10.

## 1. Giới thiệu bài toán (2 slide)
- Động lực: hồ sơ dài, bác sĩ thiếu thời gian, dễ sót thông tin.
- Sơ đồ một dòng **Input → Thuật toán → Output**:
  - Input: discharge summary bỏ mục *Hospital Course* (HPI, khám, xét nghiệm…)
  - Thuật toán: **Pointer-Generator + Coverage**
  - Output: đoạn tóm tắt *Hospital Course*
- Một ví dụ minh họa tự soạn (không dùng văn bản MIMIC thật).

## 2. Nghiên cứu liên quan (1–2 slide)
- Seq2seq + attention (Bahdanau 2015; Nallapati 2016)
- Pointer-Generator + Coverage (See và cộng sự, ACL 2017)
- Tóm tắt Hospital Course: Adams và cộng sự 2021; shared task "Discharge Me!" (BioNLP 2024)
- Transformer tiền huấn luyện (BART, T5, ClinicalBERT): nêu ngắn, là công cụ/mở rộng
- Mỗi mục có dòng "khác gì so với đề tài". **Kiểm tra lại trích dẫn trước khi đưa lên slide.**

## 3. Phương pháp: Pointer-Generator (3 slide, trọng tâm)
1. Kiến trúc: encoder BiLSTM, decoder LSTM, attention (hình vẽ).
2. Cơ chế copy: P(w) = p_gen·P_vocab(w) + (1−p_gen)·Σ attention trên các vị trí có từ w.
   Lý do: tên thuốc, liều, từ hiếm được copy từ văn bản gốc nên ít bị bịa.
3. Coverage loss để giảm lặp. Câu chốt: "NER/BERT chỉ là công cụ đánh giá, không phải phương pháp."

## 4. Thử nghiệm (4–5 slide)
- **4.1 Dữ liệu:** bảng gồm Asclepius (discharge summary tổng hợp, công khai, bộ thay thế tạm thời)
  và MIMIC-IV-Note (bộ chính khi PhysioNet duyệt). Cột: số mẫu `[kết quả]`, độ dài trung bình `[kết quả]`,
  chia train/val/test `[kết quả]`. Tiền xử lý: tách mục, cắt độ dài, xây từ điển.
- **4.2 Độ đo:** ROUGE-1/2/L; entity recall, hallucination rate; tỉ lệ lặp n-gram.
- **4.3 Kết quả:** bảng Lead-3 / Seq2seq-attn / PG / PG+coverage `[kết quả]`; một ví dụ đầu ra `[kết quả]`.
- **4.4 Đối sánh:** ablation copy và coverage; đối chiếu gián tiếp với số liệu công bố
  (See 2017, các bài Hospital Course), ghi rõ khác bộ dữ liệu.

## 5. Tiến độ và kế hoạch (1 slide)
- Đã làm: PhysioNet credentialing đã duyệt; báo cáo CITI chờ review từ 19/10; repo; môi trường GPU;
  mô hình chạy trên dữ liệu thay thế `[xác nhận sau khi session chính xong]`.
- Tiếp theo: ký DUA, chuyển sang MIMIC-IV-Note, phân tích lỗi, demo.

## 6. Câu hỏi xin ý kiến thầy (1 slide)
- Pointer-Generator làm phương pháp chính có được chấp nhận không?
- Có thêm BART làm mốc tham chiếu được không?
- Chỉ tiêu ROUGE-L ≥ 0.45 khó đạt với tóm tắt Hospital Course, có thể diễn giải lại không?
