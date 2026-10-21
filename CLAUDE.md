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

## Kết luận đã xác nhận cho Dạng 2 và Dạng 7 (sau kiểm toán nhiều vòng)

- **Dạng 2 (tách ngưỡng tự động):** Công thức `F(g)=A(g)*B(g)-1` với
  `A(g)=t(g)/(P-t(g))`, `B(g)=[m(g)-m(G-1)]²` đã được **chứng minh bằng đại số tổng quát**
  (không chỉ số liệu cụ thể) là tương đương `A(g)*B(g) ≡ ω₀·ω₁·(μ₀-μ₁)²` — chính là phương
  sai liên lớp Otsu chuẩn, chỉ lệch hằng số cộng `-1` (không đổi vị trí cực đại). Dù vậy,
  **PHẢI triển khai đúng công thức của tài liệu viết ra** (A, B, F theo đúng dạng trên,
  dùng m(G-1) là trung bình toàn cục chứ không phải μ₁ của riêng lớp trên), **không** thay
  bằng công thức Otsu "chuẩn" (`ω₀ω₁(μ₀-μ₁)²`) hay hàm có sẵn (`cv2.threshold(..., THRESH_OTSU)`)
  dù chúng tương đương toán học — vì các bước trung gian A(g), B(g) phải khớp đúng cách
  tài liệu trình bày để học sinh đối chiếu lời giải.
- **Dạng 7 (nhân chập tổng quát):** Anchor (điểm neo) đã xác nhận bằng 2 ví dụ số đầy đủ
  (BÀI 4.pdf, mặt nạ 2×2) khớp 100% với công thức 2.1 **viết nguyên văn** (`I(x+i,y+j)*T(i,j)`,
  `i,j=0..m-1`, không dịch chuyển). Đây là quy tắc áp dụng cho **Dạng 7 nói chung**.
  Tài liệu BÀI 7 (toán tử Laplace, phát hiện biên — **không thuộc 7 dạng bài, là chương khác**)
  có công thức đại số riêng `2I(x,y)-I(x-1,y)-I(x+1,y)+...` tự nhiên đối xứng quanh tâm —
  đây là đặc thù của phép đạo hàm bậc 2 Laplace, **không phải quy tắc anchor của công thức 2.1
  tổng quát**. **Không được dùng tính đối xứng của Laplace để suy ra hay thay đổi anchor của
  Dạng 7** — hai ngữ cảnh độc lập, không được trộn lẫn. Nếu cần áp dụng Dạng 7 với mặt nạ
  3×3 trở lên (chưa có ví dụ số đầy đủ để verify), dùng công thức 2.1 nguyên văn làm mặc định.

## Kiến trúc

- `algorithms/`: logic thuật toán thuần Python/NumPy, độc lập hoàn toàn với giao diện.
  Mỗi thuật toán nhận input rõ ràng (ma trận ảnh + tham số) và trả về kết quả + các bước
  tính toán trung gian (để hiển thị lên giao diện). Validation input dùng `core/validators.py`.
- `core/`: hàm tiện ích dùng chung (validators, matrix_utils) — không phải business logic
  thuật toán.
- `tests/`: pytest cho từng thuật toán, dùng ma trận tổng hợp tự tạo (đáp án tính tay độc
  lập) để có thể commit public, cộng với test đối chiếu `docs/audit/*.json` (tự động skip
  nếu không có, vì đó là dữ liệu local-only — xem "Quy tắc Git và bảo vệ dữ liệu").
- Giao diện: dự án đã chuyển từ Streamlit sang **website chạy local** (FastAPI backend +
  HTML/CSS/JS thuần, phục vụ tại `http://127.0.0.1:8000`). Streamlit không còn là giao diện
  chính của project — không thêm code Streamlit mới. Cấu trúc dự kiến: `server.py` (FastAPI
  app), `web/` (frontend tĩnh: `index.html`, `css/`, `js/`), launcher Windows (`.bat` hoặc
  `.pyw`) để mở một cú nhấp đúp mà không cần gõ lệnh terminal.

Không tạo class/abstraction/design pattern khi chưa cần thiết. Không tạo database, AI
API, OCR, hay RAG ở giai đoạn này. Không over-engineering — đây là mini project sinh viên.

## Quy tắc Git và bảo vệ dữ liệu

- **14 file PDF bài giảng trong `docs/` ĐÃ ĐƯỢC NGƯỜI DÙNG XÁC NHẬN CHO PHÉP PUBLIC** và đã
  push lên GitHub (commit `205afe1`) để phục vụ việc ChatGPT và người khác kiểm chứng độc
  lập. `.gitignore` dùng pattern `docs/*` + `!docs/*.pdf` để cho phép đúng các PDF này,
  KHÔNG áp dụng quy tắc "không bao giờ commit docs/" một cách tuyệt đối nữa.
- **Vẫn tuyệt đối không commit** `docs/audit/` (chứa golden test JSON trích xuất chi tiết
  từ bài giảng — khác về bản chất với PDF gốc đã được phép public), `data/`, `datasets/`,
  `uploads/`, `outputs/`, `local_notes/`, `.venv/`, `__pycache__/`, `.pytest_cache/`,
  `.streamlit/secrets.toml`, `.env`, log file, ảnh PNG render từ PDF, script kiểm tra tạm.
- `.gitignore` không bảo vệ file đã được track trước đó — luôn kiểm tra
  `git status --ignored` và danh sách file đang track trước khi commit, đặc biệt xác nhận
  `docs/audit/` vẫn xuất hiện trong mục "Ignored files".
- Trước mỗi commit: kiểm tra `git status`, `git diff --cached --name-only`, đảm bảo không có
  thư mục/file nhạy cảm bị staging.
- Ưu tiên `git add <path-cụ-thể>`. Không dùng `git add .` một cách mù quáng.
- Không `--force` push, không xóa branch, không ghi đè công việc người khác.

## Quy tắc commit message

- Message đơn giản, tự nhiên, mô tả đúng công việc.
- Ví dụ: `Initial project setup`, `Implement histogram equalization`, `Fix matrix calculation`.
- **Không** dùng Conventional Commits (không có tiền tố `feat:`, `fix:`, `chore:`, v.v.).

## Quy tắc auto commit và auto push

- Sau mỗi công việc được giao đã hoàn thành và kiểm tra thành công (test pass, không lỗi),
  tự động commit và push lên `origin/main` mà không cần hỏi lại.
- Trước mỗi commit: chạy `git status` và `git diff --cached --name-only` để đảm bảo không có
  file nhạy cảm hoặc dữ liệu tài liệu học bị staging.
- Không tạo commit rỗng khi không có thay đổi thực sự về source hoặc cấu hình.
- Không push code chưa kiểm tra, code lỗi, hoặc thay đổi chưa được yêu cầu.
- Không `--force` push, không xóa branch, không ghi đè công việc người khác.
- Nếu push thất bại, báo lý do và giữ nguyên trạng thái an toàn, không tự tìm cách vượt qua
  quyền truy cập.

## Phạm vi hiện tại (V1)

- Backend: Python + NumPy + FastAPI + Uvicorn + pytest. Frontend: HTML/CSS/JS thuần, không
  npm/build tool trừ khi thực sự cần thiết. Chạy local tại `http://127.0.0.1:8000`.
- Chưa tích hợp AI API, OCR, RAG, database.
- Tương lai có thể mở rộng AI đọc ảnh đề bài / RAG tra cứu kiến thức — không triển khai sớm.

## Những điểm chưa được tài liệu xác nhận (không được tự ý quyết định thay)

- Dạng 2: quy tắc tạo ảnh nhị phân sau khi tìm được ngưỡng (`>=θ` hay khác) chưa có ví dụ
  số trực tiếp — code trả về trong field riêng `unconfirmed_binary_result`, không trộn vào
  kết quả chính.
- Dạng 5: quy tắc tie-break khi 2+ giá trị khác nhau cùng khoảng cách tới pixel trung tâm —
  chưa có căn cứ, code cảnh báo qua `tie_break_used`/`tie_break_used_anywhere`.
- Dạng 7: anchor cho mẫu T kích thước lẻ (3×3 trở lên) — chưa có ví dụ số độc lập đầy đủ,
  code cảnh báo qua `anchor_rule_confirmed=False`.
- Làm tròn giá trị đúng .5 (Dạng 1, 4, 5): không có ví dụ PDF rơi đúng trường hợp này —
  `core/matrix_utils.round_half_up` dùng round-half-up làm lựa chọn hợp lý nhất hiện có.

Không xóa các cờ cảnh báo này hay "chọn đại" một quy tắc rồi báo là đáp án chắc chắn khi
chưa có bằng chứng PDF trực tiếp.
