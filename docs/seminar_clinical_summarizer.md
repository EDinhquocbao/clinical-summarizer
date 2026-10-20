# Semina đầu tiên: Tóm tắt tự động hồ sơ bệnh án điện tử (Clinical NLP)

**Môn:** Deep Learning, đề tài cuối kỳ (Dự án 29)
**Ngày semina:** 20/10/2026
**Cách dùng file này:** phần *"Có thể nói:"* là lời nói mẫu để bạn đọc to và diễn đạt lại theo cách của mình. Phần gạch đầu dòng là ý chính nếu bạn muốn nói tự nhiên không đọc.

---

## 0. Tổng quan buổi nói

**Mục tiêu:** trình bày 5 việc: (1) bài toán, (2) dữ liệu, (3) phương pháp, (4) các so sánh dự kiến, (5) tiến độ hiện tại và kế hoạch tiếp theo.

**Thời lượng gợi ý:** khoảng 12 đến 15 phút, chưa tính hỏi đáp. Nếu thầy cho ít thời gian hơn, bỏ bớt các phần có dấu (*) .

| Phần | Nội dung | Thời gian |
|---|---|---|
| 1 | Mở đầu và động lực | 1,5 phút |
| 2 | Bài toán | 2 phút |
| 3 | Dữ liệu | 3 phút |
| 4 | Phương pháp | 3,5 phút |
| 5 | Đánh giá và các so sánh (*) | 2 phút |
| 6 | Tiến độ và kế hoạch | 2 phút |
| 7 | Kết luận | 0,5 phút |

---

## 1. Mở đầu và động lực (1,5 phút)

**Ý chính**
- Hồ sơ bệnh án điện tử (EHR) chứa rất nhiều văn bản tự do do bác sĩ ghi: lý do nhập viện, diễn tiến điều trị, thuốc, kết quả xét nghiệm.
- Khi bệnh nhân xuất viện, bác sĩ phải viết **báo cáo xuất viện (discharge summary)** tổng hợp lại toàn bộ đợt điều trị. Việc này tốn thời gian, và người đọc sau (bác sĩ tuyến dưới, bác sĩ khám lại) cũng tốn thời gian để đọc.
- Nếu máy tự động tóm tắt và trích xuất được thông tin quan trọng thì có thể hỗ trợ bác sĩ tiết kiệm thời gian.
- Nhưng y tế khác các lĩnh vực khác ở chỗ: **sai một thông tin (bịa ra một loại thuốc hay liều dùng) có thể gây hại**. Đây là lý do đề tài không chỉ làm tóm tắt mà còn kết hợp trích xuất thông tin để kiểm soát.

**Có thể nói:**
> "Em xin trình bày đề tài tóm tắt tự động hồ sơ bệnh án điện tử bằng các mô hình học sâu. Khi một bệnh nhân xuất viện, bác sĩ phải viết một bản báo cáo xuất viện tổng hợp lại cả đợt điều trị, và những văn bản này thường rất dài. Việc tóm tắt tự động có thể giúp bác sĩ tiết kiệm thời gian. Tuy nhiên trong y tế, một lỗi nhỏ như bịa ra tên thuốc hay liều dùng cũng có thể nguy hiểm, nên em kết hợp thêm việc trích xuất thông tin y tế để kiểm soát chất lượng bản tóm tắt."

---

## 2. Bài toán (2 phút)

### 2.1 Phát biểu bài toán
- **Đầu vào:** văn bản hồ sơ bệnh án dài (ví dụ discharge summary), thường vài nghìn từ.
- **Đầu ra:** (1) bản tóm tắt ngắn; (2) danh sách thông tin y tế quan trọng: chẩn đoán, thuốc, liều dùng, kết quả xét nghiệm; (3) báo cáo xuất viện có cấu trúc.

### 2.2 Hai bài toán con
1. **Information Extraction (trích xuất thông tin):**
   - **NER (Named Entity Recognition):** nhận dạng và gán nhãn các thực thể y tế trong văn bản (bệnh, thuốc, liều, xét nghiệm).
   - **Relation Extraction:** tìm quan hệ giữa các thực thể (thuốc nào đi với liều nào, thuốc dùng để điều trị bệnh gì).
2. **Abstractive Summarization (tóm tắt trừu tượng):** mô hình **tự sinh** câu mới để tóm tắt, không chỉ chọn lại các câu có sẵn.

### 2.3 Ý tưởng chính của đề tài
- Hai bài toán **bổ trợ nhau**: dùng kết quả NER để **dẫn hướng** mô hình tóm tắt (giảm bỏ sót thông tin quan trọng), và dùng NER để **kiểm tra chéo** bản tóm tắt (phát hiện thuốc hoặc liều xuất hiện trong tóm tắt mà không có trong bản gốc).

### 2.4 Chỉ tiêu của đề tài
- ROUGE-L ≥ 0.45
- Trích xuất đúng ≥ 90% thực thể y tế quan trọng
- Có demo ứng dụng web

**Ví dụ minh họa (do em tự soạn, không phải dữ liệu thật):**
> *Văn bản gốc (rút gọn):* "Bệnh nhân nam 65 tuổi nhập viện vì đau ngực. Tiền sử tăng huyết áp. Chẩn đoán: nhồi máu cơ tim cấp. Điều trị: aspirin 81 mg mỗi ngày, metoprolol 25 mg hai lần mỗi ngày..."
> *Thực thể:* Bệnh = nhồi máu cơ tim cấp, tăng huyết áp; Thuốc = aspirin, metoprolol; Liều = 81 mg, 25 mg.
> *Lỗi cần phát hiện (hallucination):* nếu bản tóm tắt viết "dùng warfarin" nhưng bản gốc không có thuốc này thì phải bị cảnh báo.

**Có thể nói:**
> "Đầu vào là một văn bản hồ sơ rất dài, đầu ra gồm một bản tóm tắt ngắn cùng các thông tin quan trọng như chẩn đoán, thuốc, liều dùng. Em chia thành hai bài toán con là trích xuất thông tin và tóm tắt trừu tượng. Điểm em muốn nhấn mạnh là hai phần này bổ trợ nhau: kết quả trích xuất thực thể vừa dẫn hướng cho mô hình tóm tắt, vừa dùng để kiểm tra xem bản tóm tắt có bịa ra thông tin hay không."

---

## 3. Dữ liệu (3 phút)

### 3.1 Dữ liệu đề tài ban đầu nêu
Đề tài gốc nêu ba nguồn: **MIMIC-III Discharge Summaries**, **i2b2** và **n2c2**.

### 3.2 Các nguồn dữ liệu và vai trò

| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| **MIMIC-III** (PhysioNet) | Văn bản discharge summary để tóm tắt | Dữ liệu bệnh án đã khử định danh của các bệnh nhân điều trị tại một bệnh viện ở Boston |
| **MIMIC-IV-Note** (PhysioNet) | Phương án thay thế/bổ sung cho MIMIC-III | Chứa discharge summary và báo cáo X-quang; cần xác nhận chọn bộ nào làm chính |
| **MIMIC-IV-Ext-EntityCoding** (PhysioNet) | NER và assertion trên discharge summary | Theo trang dữ liệu: 400 discharge summary, 85.328 chú thích thực thể, 10.922 phân loại assertion |
| **BC5CDR, NCBI-Disease** (công khai) | NER thuốc/hóa chất và bệnh có nhãn chuẩn | Là văn bản y văn (tóm tắt bài báo PubMed), không phải hồ sơ bệnh án |
| **n2c2 / i2b2** | Dữ liệu NER và quan hệ nổi tiếng trong Clinical NLP | **Hiện tạm không khả dụng** (xem 3.3) |
| **MIMIC-III-Ext-VeriFact-BHC** (PhysioNet) | Tham khảo cách đánh giá hallucination | Mới xem mô tả, chưa nghiên cứu kỹ |
| Nhãn tự động trên MIMIC | Bổ sung nhãn thuốc, liều, xét nghiệm | Dùng công cụ như scispaCy/MedCAT; nhãn không hoàn hảo nên sẽ kiểm tra thủ công một mẫu nhỏ |

### 3.3 Tình hình n2c2 / i2b2 (cần nói rõ và trung thực)
- Các bộ dữ liệu n2c2 (tiếp nối i2b2) được phát hành qua cổng DBMI của Đại học Harvard. Hiện cổng ghi bộ dữ liệu **"tạm thời không khả dụng"** và không mở đăng ký. Trang chưa nêu lý do cũng như thời điểm mở lại.
- Các track 2022 đã đóng đăng ký; các track 2019 phát hành qua Mayo Clinic không phù hợp đề tài (về độ tương đồng văn bản và tiền sử gia đình).
- Em đã gửi email hỏi qua form liên hệ của n2c2 và đang chờ phản hồi.
- **Hướng xử lý:** coi n2c2 là nguồn bổ trợ, không bắt buộc; thay bằng EntityCoding, BC5CDR/NCBI-Disease và nhãn tự động. Nếu n2c2 mở lại thì bổ sung.
- Thông tin thêm có ích: bộ n2c2 2018 Track 2 (trích xuất thuốc và biến cố bất lợi của thuốc) gồm 505 discharge summary **lấy từ chính MIMIC-III**, với 9 loại thực thể (ADE, Dosage, Duration, Drug, Form, Frequency, Reason, Route, Strength). Vì vậy kết quả công bố trên bộ này vẫn dùng được làm **mốc tham khảo** cho báo cáo (hệ thống tốt nhất đạt F1 khoảng 0,94 cho trích xuất khái niệm), dù em không có dữ liệu để tái lập.

### 3.4 Quy trình xin quyền truy cập và đạo đức dữ liệu
- Dữ liệu MIMIC là dữ liệu lâm sàng thật (đã khử định danh), nên **không tải tự do**. Cần: tài khoản PhysioNet, hồ sơ credentialing (có người tham chiếu là giảng viên), khóa đào tạo đạo đức nghiên cứu CITI, rồi ký thỏa thuận sử dụng dữ liệu (DUA).
- Quy tắc phải tuân thủ: không chia sẻ dữ liệu, mỗi người tự xin quyền, **không đưa văn bản bệnh án thật vào slide, báo cáo công khai hay demo**, và **không dán dữ liệu vào dịch vụ mô hình ngôn ngữ bên thứ ba** (PhysioNet có chính sách riêng về việc này).
- Vì vậy demo sẽ dùng văn bản mẫu tự soạn.

**Có thể nói:**
> "Về dữ liệu, đề tài gốc nêu MIMIC-III, i2b2 và n2c2. Dữ liệu y tế là dữ liệu nhạy cảm nên phải xin quyền truy cập. Với MIMIC em đã được PhysioNet duyệt hồ sơ và đang chờ duyệt báo cáo đào tạo để ký thỏa thuận sử dụng. Riêng n2c2, hiện cổng của Harvard đang báo dữ liệu tạm thời không khả dụng, em đã gửi email hỏi nhưng chưa có phản hồi. Em không muốn bị phụ thuộc vào nó nên đã tìm các nguồn thay thế: bộ EntityCoding trên PhysioNet có nhãn thực thể trên discharge summary, cùng các bộ công khai như BC5CDR và NCBI-Disease, và bổ sung nhãn tự động trên MIMIC. Kết quả công bố của n2c2 em vẫn dùng làm mốc so sánh trong tài liệu."

---

## 4. Phương pháp (3,5 phút)

### 4.1 Pipeline tổng thể

```
Hồ sơ bệnh án (văn bản thô)
        │
        ▼
[1] Tiền xử lý: xử lý ký hiệu ẩn danh, tách các mục (section)
        │
        ▼
[2] NER / Relation Extraction  (BioBERT / ClinicalBERT)
        │
        ▼
[3] Tóm tắt có điều kiện  (BART / T5 / LED + danh sách thực thể)
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

### 4.2 Từng khối, vì sao chọn mô hình này

**Tiền xử lý.** Dữ liệu đã khử định danh nên chứa các ký hiệu thay thế cho tên, ngày, địa chỉ; cần xử lý nhất quán. Văn bản discharge summary có các mục như lý do nhập viện, diễn tiến điều trị, thuốc khi xuất viện; có thể tách để xử lý riêng.

**NER và Relation Extraction: BioBERT, ClinicalBERT.**
- Cả hai là mô hình dạng **encoder** (họ BERT), phù hợp cho bài toán gán nhãn từng token.
- **BioBERT** được huấn luyện thêm trên văn bản y sinh (bài báo PubMed). **ClinicalBERT** được huấn luyện thêm trên ghi chú lâm sàng của MIMIC nên gần với loại văn bản của đề tài hơn.
- Cách làm: fine-tune dạng *token classification* (mỗi từ được gán nhãn B/I/O theo từng loại thực thể).

**Tóm tắt: BART, T5, LED.**
- Đây là các mô hình **encoder-decoder** (seq2seq), phù hợp để sinh văn bản.
- **BART** được huấn luyện theo kiểu khôi phục văn bản bị làm hỏng, mạnh cho tóm tắt; **T5** xem mọi bài toán là "văn bản vào, văn bản ra", linh hoạt.
- **Vấn đề kỹ thuật chính: văn bản quá dài.** BART chỉ nhận tối đa 1024 token, trong khi discharge summary thường dài hơn nhiều. Ba hướng xử lý:
  1. **Chia đoạn rồi tóm tắt phân cấp:** tóm tắt từng đoạn, sau đó tóm tắt lại các bản tóm tắt.
  2. **Dùng LED (Longformer-Encoder-Decoder):** dùng cơ chế attention thưa nên nhận được đầu vào rất dài (tới khoảng 16 nghìn token).
  3. **Tóm tắt theo từng mục** (section) rồi ghép lại.

**Dẫn hướng bằng thực thể (ý tưởng cốt lõi).** Đưa danh sách thực thể quan trọng do NER trích ra vào đầu vào của mô hình tóm tắt, để mô hình biết "phải giữ lại những gì".

**Hậu kiểm.** Chạy NER trên bản tóm tắt sinh ra, đối chiếu với thực thể của văn bản gốc; thực thể nào xuất hiện trong tóm tắt mà không có trong bản gốc sẽ bị đánh dấu nghi ngờ hallucination.

### 4.3 Thiết lập huấn luyện (dự kiến)
- Fine-tune các mô hình tiền huấn luyện sẵn (không huấn luyện từ đầu).
- Nếu thiếu bộ nhớ GPU: dùng bản *base*, huấn luyện độ chính xác hỗn hợp (mixed precision), hoặc LoRA.
- Chia train/val/test **theo bệnh nhân** để tránh rò rỉ dữ liệu giữa các tập.

**Có thể nói:**
> "Pipeline của em có sáu khối. Đầu tiên là tiền xử lý. Sau đó là trích xuất thực thể bằng BioBERT hoặc ClinicalBERT. ClinicalBERT được huấn luyện thêm trên ghi chú lâm sàng của MIMIC nên rất gần với dữ liệu của đề tài. Khối tiếp theo là tóm tắt bằng các mô hình sinh văn bản như BART, T5 hoặc LED. Thách thức kỹ thuật lớn nhất là văn bản rất dài, trong khi BART chỉ nhận khoảng một nghìn token. Em dự kiến thử ba cách: chia đoạn rồi tóm tắt phân cấp, dùng LED vốn nhận được đầu vào dài, hoặc tóm tắt theo từng mục. Điểm cốt lõi là em đưa danh sách thực thể quan trọng vào đầu vào của mô hình tóm tắt, và sau cùng dùng NER để kiểm tra chéo bản tóm tắt, phát hiện thông tin bị bịa."

---

## 5. Đánh giá và các so sánh (2 phút) (*)

### 5.1 Các chỉ số đánh giá
| Phần | Chỉ số | Ý nghĩa |
|---|---|---|
| NER | Precision, Recall, F1 **theo từng loại thực thể** | Đo độ chính xác trích xuất; báo cáo riêng từng loại vì thuốc dễ hơn chẩn đoán |
| Tóm tắt | ROUGE-1/2/L | Đo độ trùng lặp từ/cụm từ với bản tóm tắt tham chiếu |
| Tóm tắt | BERTScore | Đo tương đồng về ngữ nghĩa bằng embedding |
| An toàn | **Entity recall** | Bao nhiêu thực thể quan trọng của bản gốc còn xuất hiện trong tóm tắt |
| An toàn | **Hallucination rate** | Tỉ lệ thuốc/liều/chẩn đoán xuất hiện trong tóm tắt nhưng không có trong bản gốc |
| Thủ công | Đánh giá trên mẫu nhỏ | Kiểm tra cảm quan và độ an toàn |

### 5.2 Các so sánh dự kiến
1. **Baseline và mô hình học sâu:** Lead-3 (chọn 3 câu đầu) và BART chưa fine-tune so với mô hình đã fine-tune. Cho thấy học sâu mang lại cải thiện bao nhiêu.
2. **Tóm tắt thuần so với tóm tắt có dẫn hướng bằng thực thể (ablation):** đây là so sánh trọng tâm, kiểm chứng ý tưởng của đề tài.
3. **Các kiến trúc tóm tắt:** BART, T5, LED; và các cách xử lý văn bản dài (phân cấp, LED, theo mục).
4. **BioBERT và ClinicalBERT** cho NER: mô hình tiền huấn luyện trên y văn so với trên ghi chú lâm sàng.
5. **So sánh nguồn dữ liệu NER:** nhãn chuẩn (EntityCoding, BC5CDR) và nhãn tự động; kết quả trên y văn và trên văn bản lâm sàng thật.
6. **So sánh với tài liệu:** đối chiếu với kết quả công bố trên n2c2 2018 Track 2 (F1 tốt nhất khoảng 0,94 cho trích xuất khái niệm; các loại như ADE và Reason khó hơn). Cần lưu ý đây là so sánh gián tiếp vì không cùng bộ dữ liệu.

### 5.3 Nhận định trung thực về chỉ tiêu
- **ROUGE-L ≥ 0.45** có thể khó đạt vì kết quả phụ thuộc rất nhiều vào việc **định nghĩa "bản tóm tắt chuẩn"** (ví dụ lấy mục *Brief Hospital Course* làm đích). Em sẽ nêu rõ định nghĩa này, và bổ sung BERTScore, entity recall thay vì chỉ dựa vào ROUGE.
- **Trích xuất đúng ≥ 90%** có khả năng đạt với thực thể đơn giản như tên thuốc, nhưng khó hơn với chẩn đoán và các quan hệ; vì vậy báo cáo theo từng loại thực thể.

**Có thể nói:**
> "Em đánh giá bằng ba nhóm chỉ số: F1 theo từng loại thực thể cho phần trích xuất, ROUGE và BERTScore cho phần tóm tắt, và hai chỉ số về an toàn là entity recall và tỉ lệ hallucination. So sánh quan trọng nhất là giữa tóm tắt thuần và tóm tắt có dẫn hướng bằng thực thể, để kiểm chứng ý tưởng chính của đề tài. Ngoài ra em so sánh BioBERT với ClinicalBERT, các mô hình tóm tắt BART, T5, LED, và đối chiếu với kết quả đã công bố trên n2c2. Em cũng nhận thấy chỉ tiêu ROUGE-L 0.45 phụ thuộc nhiều vào cách định nghĩa bản tóm tắt chuẩn nên em sẽ nêu rõ trong báo cáo."

---

## 6. Tiến độ hiện tại và kế hoạch (2 phút)

### 6.1 Đã làm được
- [x] Phân tích đề tài, xác định hai bài toán con và ý tưởng chính.
- [x] Lập kế hoạch 7 giai đoạn.
- [x] **PhysioNet:** nộp hồ sơ credentialing ngày 13/10, đã được duyệt ngày 18/10/2026.
- [x] **Khóa CITI:** hoàn thành cả hai khóa (Conflicts of Interest và Data or Specimens Only Research); đã nộp báo cáo đào tạo lên PhysioNet ngày 19/10, **đang chờ duyệt**.
- [x] **n2c2:** tạo tài khoản, phát hiện dữ liệu tạm không khả dụng, đã gửi email hỏi.
- [x] Khảo sát và đề xuất các nguồn dữ liệu thay thế trên PhysioNet và nguồn công khai.

### 6.2 Chưa làm (nói thẳng)
- Chưa có dữ liệu MIMIC (chờ duyệt đào tạo, ký DUA và tải).
- Chưa viết code; **chưa có kết quả thực nghiệm nào**.

### 6.3 Kế hoạch tiếp theo
| Bước | Việc | Ghi chú |
|---|---|---|
| 1 | Chờ duyệt đào tạo, ký DUA, tải dữ liệu | Song song làm bước 2 |
| 2 | Dựng môi trường, chạy thử NER trên BC5CDR/NCBI-Disease | Không phụ thuộc quyền truy cập |
| 3 | Tiền xử lý và khảo sát dữ liệu (EDA) | Phân bố độ dài, các mục |
| 4 | Baseline (Lead-3, BART chưa fine-tune) | Lấy điểm mốc |
| 5 | Huấn luyện NER (BioBERT, ClinicalBERT) | F1 theo từng loại |
| 6 | Huấn luyện tóm tắt thuần và có dẫn hướng | Ablation |
| 7 | Đánh giá, phân tích lỗi, hallucination | |
| 8 | Web demo và báo cáo | |

### 6.4 Điều em cần ở thầy (nếu muốn hỏi ý kiến)
- Thống nhất chọn **MIMIC-III hay MIMIC-IV-Note** làm dữ liệu chính (hoặc dùng cả hai: một bộ chính, một bộ làm tập kiểm tra bên ngoài).
- Xác nhận cách diễn giải chỉ tiêu theo bộ dữ liệu thực tế do n2c2 không khả dụng.
- Thời hạn cụ thể và yêu cầu của môn về báo cáo/demo.

**Có thể nói:**
> "Về tiến độ, em đã hoàn thành phần phân tích và lập kế hoạch, và đã xong các thủ tục xin quyền truy cập: hồ sơ PhysioNet đã được duyệt, em đã hoàn thành khóa đào tạo CITI và nộp báo cáo, hiện đang chờ duyệt để ký thỏa thuận sử dụng dữ liệu. Em xin nói thẳng là hiện em chưa có dữ liệu thật và chưa có kết quả thực nghiệm. Trong lúc chờ, em sẽ dựng môi trường và chạy thử phần NER trên dữ liệu công khai để không mất thời gian. Có hai điểm em muốn xin ý kiến thầy: chọn MIMIC-III hay MIMIC-IV-Note làm dữ liệu chính, và cách diễn giải chỉ tiêu khi n2c2 chưa khả dụng."

---

## 7. Kết luận (0,5 phút)

**Có thể nói:**
> "Tóm lại, đề tài kết hợp trích xuất thông tin và tóm tắt trừu tượng để vừa giúp bác sĩ tiết kiệm thời gian, vừa kiểm soát được nguy cơ thông tin bị bịa. Em đã hoàn tất phần xin quyền dữ liệu và có phương án thay thế khi n2c2 chưa khả dụng. Phần tiếp theo em sẽ bắt tay vào xây dựng pipeline và thực nghiệm. Em xin cảm ơn thầy và các bạn đã lắng nghe."

---

## Phụ lục A. Câu hỏi có thể gặp và gợi ý trả lời

**1. Tại sao không dùng thẳng một mô hình ngôn ngữ lớn như ChatGPT để tóm tắt?**
- Dữ liệu MIMIC có điều kiện sử dụng nghiêm ngặt; PhysioNet có hướng dẫn riêng về việc dùng mô hình ngôn ngữ lớn với dữ liệu này nên không đưa dữ liệu thật vào các dịch vụ bên ngoài.
- Môn học yêu cầu tự huấn luyện/fine-tune mô hình học sâu; mô hình nhỏ chạy cục bộ dễ kiểm soát, dễ đo lường và phân tích lỗi hơn.

**2. Đã có tóm tắt rồi, sao còn cần NER?**
- NER giúp tạo báo cáo có cấu trúc (chẩn đoán, thuốc, xét nghiệm), dẫn hướng mô hình tóm tắt để giảm bỏ sót, và dùng để kiểm tra chéo phát hiện hallucination.

**3. Hallucination là gì và sao đo được?**
- Là khi mô hình sinh ra thông tin không có trong văn bản gốc (ví dụ một loại thuốc không từng được kê). Em đo bằng cách trích thực thể từ bản tóm tắt rồi đối chiếu với bản gốc, tính tỉ lệ thực thể "lạ".

**4. ROUGE-L 0.45 có đạt được không?**
- Em chưa có kết quả nên chưa thể khẳng định. Chỉ tiêu này phụ thuộc cách định nghĩa bản tóm tắt chuẩn, nên em sẽ nêu rõ thiết lập và báo cáo thêm các chỉ số khác.

**5. Tại sao đổi khỏi n2c2?**
- Không phải em chủ động đổi, mà dữ liệu hiện tạm không khả dụng trên cổng của Harvard. Em đã gửi email hỏi và coi n2c2 là nguồn bổ trợ; nếu mở lại sẽ bổ sung.

**6. Dữ liệu không đủ nhãn thì sao?**
- Kết hợp nhãn chuẩn (EntityCoding, BC5CDR/NCBI-Disease) và nhãn tự động, rồi kiểm tra thủ công một mẫu nhỏ để ước lượng độ chính xác; báo cáo rõ đâu là nhãn tự động.

**7. Vì sao chọn ClinicalBERT và BioBERT?**
- BioBERT mạnh trên y văn; ClinicalBERT được huấn luyện thêm trên ghi chú lâm sàng MIMIC nên gần domain hơn. Em sẽ so sánh trực tiếp hai mô hình.

**8. Văn bản dài thì xử lý thế nào?**
- Ba hướng: chia đoạn tóm tắt phân cấp, dùng LED nhận đầu vào dài, hoặc tóm tắt theo từng mục.

**9. Cần tài nguyên tính toán thế nào?**
- Em dự kiến dùng bản *base*, mixed precision và LoRA nếu cần; có thể dùng Colab/Kaggle. (Bạn thay bằng GPU thực tế của mình nếu có.)

**10. Hệ thống có thay được bác sĩ không?**
- Không. Hệ thống chỉ hỗ trợ; bác sĩ vẫn phải kiểm tra. Em sẽ ghi rõ giới hạn và rủi ro trong báo cáo.

**11. Kết quả bây giờ ra sao?**
- Chưa có kết quả thực nghiệm. Hiện đã xong thủ tục dữ liệu và kế hoạch; bước tiếp theo là dựng pipeline.

---

## Phụ lục B. Giải thích nhanh các thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| EHR | Hồ sơ bệnh án điện tử |
| Discharge summary | Báo cáo xuất viện, tổng hợp đợt điều trị |
| NER | Nhận dạng thực thể có tên (tìm và gán nhãn bệnh, thuốc... trong văn bản) |
| Relation Extraction | Trích xuất quan hệ giữa các thực thể |
| Assertion | Trạng thái của thực thể (có, không có, nghi ngờ...) |
| Abstractive summarization | Tóm tắt trừu tượng: sinh câu mới thay vì chọn lại câu có sẵn |
| Encoder / Encoder-decoder | Kiến trúc chỉ đọc hiểu (BERT) / kiến trúc đọc hiểu rồi sinh văn bản (BART, T5) |
| Fine-tune | Huấn luyện tiếp mô hình đã tiền huấn luyện trên dữ liệu của bài toán |
| ROUGE | Chỉ số đo độ trùng lặp từ/cụm từ giữa tóm tắt và tham chiếu |
| BERTScore | Chỉ số đo tương đồng ngữ nghĩa dựa trên embedding |
| Hallucination | Mô hình sinh thông tin không có trong văn bản gốc |
| De-identification | Khử định danh: che thông tin cá nhân của bệnh nhân |
| DUA | Thỏa thuận sử dụng dữ liệu |
| CITI | Chương trình đào tạo đạo đức nghiên cứu |

---

## Phụ lục C. Gợi ý bộ slide (nếu cần làm nhanh)

1. Tiêu đề và đề tài
2. Động lực: vì sao cần tóm tắt hồ sơ bệnh án
3. Bài toán: đầu vào, đầu ra, hai bài toán con (kèm ví dụ minh họa tự soạn)
4. Dữ liệu và quy trình truy cập; tình hình n2c2
5. Pipeline và các mô hình
6. Đánh giá và các so sánh dự kiến
7. Tiến độ và kế hoạch
8. Câu hỏi/ý kiến thầy

*Mỗi slide nên ít chữ, dùng sơ đồ pipeline ở mục 4.1 làm slide chính của phần phương pháp.*

---

## Phụ lục D. Lưu ý khi trình bày

- **Không đưa văn bản bệnh án thật lên slide.** Chỉ dùng ví dụ tự soạn như ở mục 2.
- **Nói rõ cái gì đã làm, cái gì là dự kiến.** Hiện chưa có kết quả thực nghiệm; đừng để người nghe hiểu nhầm rằng đã có số liệu.
- **Các con số lấy từ nguồn ngoài** (số lượng ghi chú của từng bộ dữ liệu, F1 của n2c2 2018 Track 2) là thông tin từ các trang dữ liệu và bài báo công bố; nếu bị hỏi nguồn, bạn nói là lấy từ trang PhysioNet và bài tổng quan n2c2 2018 Track 2 (Henry và cộng sự, JAMIA 2020).
- **Về chỉ tiêu ROUGE-L:** không nên khẳng định chắc chắn đạt hay không đạt khi chưa có thực nghiệm.
- **Chốt trước với thầy:** MIMIC-III hay MIMIC-IV-Note là dữ liệu chính.
