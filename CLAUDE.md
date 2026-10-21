# CLAUDE.md — Image Processing Solver

## Mục tiêu dự án

Giải 7 dạng bài toán xử lý ảnh (ma trận) đúng theo công thức và phương pháp giảng viên dạy
trong tài liệu `docs/`. Không dùng AI, OCR, RAG trong giai đoạn này.

## Nguyên tắc tối quan trọng: không tự đoán công thức

- Chỉ triển khai thuật toán theo đúng công thức, quy tắc làm tròn, quy tắc xử lý biên, và
  quy tắc so sánh ngưỡng đã được xác nhận từ tài liệu PDF trong `docs/`.
- **Không** thay thế bằng công thức "chuẩn" phổ biến trong OpenCV/NumPy/SciPy hoặc Internet
  nếu nó khác với tài liệu giảng viên — ngay cả khi công thức đó phổ biến hơn hoặc "đúng" hơn
  về mặt lý thuyết tổng quát.
- Nếu tài liệu có nhiều biến thể của cùng một thuật toán (ví dụ nhiều phiên bản lọc trung
  bình), phải phân biệt rõ biến thể nào đang được dùng và vì sao.
- Nếu tài liệu thiếu định nghĩa hoặc mâu thuẫn, phải báo cáo rõ, không tự điền bằng giả định.
- Không triển khai bất kỳ thuật toán nào trong 7 dạng bài khi chưa có xác nhận rõ ràng về
  công thức và phương pháp giải từ tài liệu.

## Kiến trúc

- `algorithms/`: logic thuật toán thuần Python/NumPy, độc lập hoàn toàn với giao diện.
  Mỗi thuật toán nhận input rõ ràng (ma trận ảnh + tham số) và trả về kết quả + các bước
  tính toán trung gian (để hiển thị lên giao diện).
- `core/`: hàm tiện ích dùng chung (không phải business logic thuật toán).
- `app.py`: giao diện Streamlit, chỉ gọi vào `algorithms/`, không chứa logic tính toán.
- `tests/`: unit test cho từng thuật toán, dùng ma trận nhỏ dễ kiểm tra bằng tay.

Không tạo class/abstraction/design pattern khi chưa cần thiết. Không tạo backend/API.
Không over-engineering — đây là mini project sinh viên.

## Quy tắc Git và bảo vệ dữ liệu

- **Không bao giờ commit** nội dung trong `docs/`, `data/`, `datasets/`, `uploads/`,
  `outputs/`, `local_notes/`, file PDF, file ảnh dữ liệu học tập, `.venv/`, `__pycache__/`,
  `.pytest_cache/`, `.streamlit/secrets.toml`, `.env`.
- `.gitignore` không bảo vệ file đã được track trước đó — luôn kiểm tra
  `git status` và danh sách file đang track trước khi commit.
- Trước mỗi commit: kiểm tra `git status`, `git diff --cached --name-only`, đảm bảo không có
  thư mục/file nhạy cảm bị staging.
- Ưu tiên `git add <path-cụ-thể>`. Không dùng `git add .` một cách mù quáng.
- Không `--force` push, không xóa branch, không ghi đè công việc người khác.

## Quy tắc commit message

- Message đơn giản, tự nhiên, mô tả đúng công việc.
- Ví dụ: `Initial project setup`, `Implement histogram equalization`, `Fix matrix calculation`.
- **Không** dùng Conventional Commits (không có tiền tố `feat:`, `fix:`, `chore:`, v.v.).

## Phạm vi hiện tại (V1)

- Chỉ Python + NumPy + Streamlit + pytest.
- Chưa tích hợp AI API, OCR, RAG, database, hoặc backend phức tạp.
- Tương lai có thể mở rộng AI đọc ảnh đề bài / RAG tra cứu kiến thức — không triển khai sớm.
