# Kết quả sơ bộ (26/10/2026)

Dữ liệu **tổng hợp** Asclepius (task Summarization), 500 mẫu đầu của tập test. Đây là chạy thử pipeline
trong lúc chờ MIMIC, không phải kết quả cuối.

## Thiết lập
- Đầu vào: yêu cầu tóm tắt + ghi chú, cắt 512 token; đích cắt 130 token; từ điển 30.000 từ (chữ thường).
- Embedding 128, hidden 256, Adam 1e-3, batch 32, dropout 0.2, clip 2.0; RTX 3050 Laptop 6GB.
- Pointer-Generator: 45 phút (2.491 bước, ~5 epoch), best val NLL 1,245.
- PG + coverage: thêm 10 phút (360 bước, λ = 1, lr 5e-4) từ checkpoint PG, val NLL 1,267.
- Seq2seq + attention: 30 phút (1.500 bước, chạy chung GPU nên chậm hơn), best val NLL 2,935
  (NLL tính trên đích có `<unk>`, không so trực tiếp với PG).
- Giải mã: beam 4, độ dài 40–140 token.

## Bảng kết quả (F1 %, trừ cột độ dài)

| Mô hình | R-1 | R-2 | R-L | Độ dài | Novel 2-gram | Lặp 3-gram ↓ | Giữ số ↑ | Bịa số ↓ | p_gen |
|---|---|---|---|---|---|---|---|---|---|
| Chỉ câu yêu cầu | 34,43 | 18,54 | 25,78 | 31,1 | 0,0 | 0,13 | 31,49 | 0,0 | - |
| Lead-3 | 32,32 | 14,85 | 22,44 | 69,6 | 0,4 | 4,69 | 40,52 | 0,0 | - |
| Oracle trích chọn (cận trên) | 74,28 | 57,67 | 58,48 | 89,0 | 2,12 | 2,25 | 75,22 | 0,82 | - |
| Seq2seq + attention | 39,64 | 18,90 | 30,47 | 74,4 | 45,52 | 16,19 | 16,14 | 53,83 | - |
| Pointer-Generator | 57,65 | 40,07 | 47,90 | 84,0 | 6,21 | 10,78 | 64,22 | 0,0 | 0,293 |
| **PG + coverage** | **58,87** | **41,19** | **48,70** | 81,4 | 6,08 | **3,69** | **65,69** | **0,0** | 0,282 |

Độ dài tham chiếu trung bình 89 token. Hình: `outputs/figures/rouge.png`, `safety.png`, `val_nll.png`.

## Nhận xét
1. **Copy là yếu tố chính:** +17,4 ROUGE-L; tỉ lệ số liệu bịa từ 53,8% về 0%; giữ số liệu từ 16% lên 64%.
2. **Coverage giảm lặp** 10,8% → 3,7% và còn tăng nhẹ ROUGE (+0,8 R-L).
3. PG + coverage đạt 83% mức Oracle trích chọn, hơn Lead-3 26 điểm ROUGE-L.
4. p_gen ≈ 0,28: mô hình chủ yếu chép, khớp với việc tham chiếu khá trích chọn (novel 2-gram chỉ ~6%).

## Hạn chế cần nói rõ
- Tóm tắt tham chiếu do GPT-3.5 viết, khá trích chọn; ghi chú ngắn hơn MIMIC nhiều → bài toán dễ hơn thực tế.
- Câu yêu cầu thường nêu sẵn chẩn đoán: riêng nó đã cho 25,8 ROUGE-L.
- Seq2seq được huấn luyện ít bước hơn PG; cần chạy lại với cùng số bước để đối sánh công bằng.
- "Bịa số" chỉ xét con số; chưa đo tên thuốc/bệnh (cần NER làm công cụ).
- Đánh giá trên 500 mẫu, một lần chạy (một seed).
