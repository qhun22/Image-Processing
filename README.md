# Image Processing Solver

Mini project giải các bài toán xử lý ảnh dạng ma trận bằng thuật toán Python thuần (không dùng AI).

Luồng xử lý: **INPUT → CHỌN THUẬT TOÁN → TÍNH TOÁN → OUTPUT**

## Trạng thái hiện tại

- ✅ **Algorithm Engine**: cả 7 thuật toán đã triển khai và kiểm thử (`algorithms/`).
- ⏳ **Website local (FastAPI + HTML/CSS/JS)**: chưa triển khai — đây là kế hoạch Mốc 2-5
  tiếp theo của dự án. Giao diện Streamlit cũ (`app.py`) không còn được coi là giao diện
  chính thức của project.

## 7 dạng bài toán

1. Cân bằng histogram (ảnh `I`, `new_level`) — `algorithms/histogram.py`
2. Tách ngưỡng tự động (ảnh `I`, `G`, `P` tùy chọn) — `algorithms/auto_threshold.py`
3. Lọc trung vị (ảnh `I`, cửa sổ `W`, ngưỡng `q`) — `algorithms/median_filter.py`
4. Lọc trung bình (ảnh `I`, cửa sổ `W`, ngưỡng `q`) — `algorithms/mean_filter.py`
5. Lọc trung bình theo k giá trị gần nhất (ảnh `I`, cửa sổ `W`, ngưỡng `q`, `k`) — `algorithms/k_nearest_mean.py`
6. Phát hiện biên Sobel (ảnh `I`, ngưỡng `q`) — `algorithms/sobel.py`
7. Nhân chập (ảnh `I`, mẫu `T`) — `algorithms/convolution.py`

## Những điểm chưa được tài liệu giảng viên xác nhận đầy đủ

Các thuật toán trên đã được đối chiếu với tài liệu PDF trong `docs/` và golden test local,
nhưng một số quy tắc chi tiết chưa có bằng chứng trực tiếp từ giáo trình. Khi gặp các
trường hợp này, hàm trả về thêm cờ cảnh báo thay vì âm thầm chọn một đáp án:

- **Dạng 2**: quy tắc tạo ảnh nhị phân sau khi tìm ngưỡng — kết quả nằm ở field riêng
  `unconfirmed_binary_result`, không trộn vào kết quả chính.
- **Dạng 5**: quy tắc xử lý khi 2+ giá trị khác nhau cùng khoảng cách tới pixel trung tâm
  (tie-break) — cờ `tie_break_used` / `tie_break_used_anywhere`.
- **Dạng 6**: tại ví dụ 7×7 chuẩn, một ô có sai lệch giữa công thức tính đúng (22) và bảng
  tổng hợp trong tài liệu gốc (26) — đây là lỗi của chính tài liệu, đã ghi chú trong
  `source_data_note`, không ảnh hưởng tới phân loại nhị phân cuối cùng.
- **Dạng 7**: anchor (điểm neo) cho mẫu T kích thước lẻ (3×3 trở lên) — cờ
  `anchor_rule_confirmed=False`. Mẫu kích thước chẵn (2×2) đã được xác nhận đầy đủ.
- Quy tắc làm tròn khi kết quả đúng `.5` (Dạng 1, 4, 5): không có ví dụ PDF rơi đúng trường
  hợp này; mặc định dùng round-half-up (`core/matrix_utils.round_half_up`), không phải
  `round()` mặc định của Python (vốn dùng banker's rounding).

## Công nghệ

- Python 3.11+
- NumPy
- pytest (kiểm thử)
- (Kế hoạch) FastAPI + Uvicorn cho backend, HTML/CSS/JS thuần cho frontend

## Cài đặt trên Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy kiểm thử

```bash
pytest
```

Test được chia 2 nhóm:
- Test dùng ma trận tổng hợp tự tạo (đáp án tính tay độc lập) — luôn chạy được, an toàn để
  public trên GitHub.
- Test đối chiếu `docs/audit/*.json` (golden test trích từ bài giảng) — tự động bỏ qua
  (`skip`) nếu thư mục này không tồn tại, vì đây là dữ liệu **chỉ dùng local, không commit**.

## Cấu trúc project

```
Image-Processing/
├── algorithms/       # 7 thuật toán xử lý ảnh, độc lập với giao diện
│   ├── histogram.py
│   ├── auto_threshold.py
│   ├── median_filter.py
│   ├── mean_filter.py
│   ├── k_nearest_mean.py
│   ├── sobel.py
│   └── convolution.py
├── core/             # Validators và tiện ích ma trận dùng chung
│   ├── validators.py
│   └── matrix_utils.py
├── tests/            # pytest cho từng thuật toán
├── docs/             # 14 PDF bài giảng (đã được phép public) + docs/audit/ (local-only)
├── app.py            # Giao diện Streamlit cũ, không còn là giao diện chính
└── requirements.txt
```

## Giới hạn đã biết

- Chưa có website/API chạy được — chỉ mới có Algorithm Engine (các hàm Python thuần).
- Xem mục "Những điểm chưa được tài liệu giảng viên xác nhận đầy đủ" ở trên.
