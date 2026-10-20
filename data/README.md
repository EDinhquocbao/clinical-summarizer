# Thư mục dữ liệu

**KHÔNG commit bất cứ file dữ liệu nào lên Git/GitHub.** Dữ liệu MIMIC và các bộ dữ liệu lâm sàng
chịu ràng buộc của Data Use Agreement: không chia sẻ, không đăng công khai.
`.gitignore` đã chặn các thư mục dưới đây, nhưng hãy tự kiểm tra bằng `git status` trước mỗi lần commit.

| Thư mục | Dùng để |
|---|---|
| `raw/` | File gốc tải từ PhysioNet (giữ nguyên, không sửa) |
| `interim/` | Kết quả trung gian (đã làm sạch, đã tách mục) |
| `processed/` | Dữ liệu sẵn sàng huấn luyện (train/val/test chia theo bệnh nhân) |
| `external/` | Dữ liệu công khai (BC5CDR, NCBI-Disease) |
