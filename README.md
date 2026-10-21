# Image Processing Solver

Mini project giải các bài toán xử lý ảnh dạng ma trận bằng thuật toán Python thuần (không dùng AI).

Luồng xử lý: **INPUT → CHỌN THUẬT TOÁN → TÍNH TOÁN → OUTPUT**

## 7 dạng bài toán

1. Cân bằng histogram (ảnh `I`, `new_level`)
2. Tách ngưỡng tự động (ảnh `I`, `G`)
3. Lọc trung vị (ảnh `I`, cửa sổ `W`, ngưỡng `q`)
4. Lọc trung bình (ảnh `I`, cửa sổ `W`, ngưỡng `q`)
5. Lọc trung bình theo k giá trị gần nhất (ảnh `I`, cửa sổ `W`, ngưỡng `q`, `k`)
6. Phát hiện biên Sobel (ảnh `I`, ngưỡng `q`)
7. Nhân chập (ảnh `I`, mẫu `T`)

## Công nghệ

- Python 3.11+
- NumPy
- Streamlit (giao diện)
- pytest (kiểm thử)

## Cài đặt trên Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Chạy kiểm thử

```bash
pytest
```

## Cấu trúc project

```
Image-Processing/
├── algorithms/   # Các thuật toán xử lý ảnh (độc lập với giao diện)
├── core/         # Các hàm dùng chung (đọc/ghi ma trận, tiện ích)
├── tests/        # Unit test cho từng thuật toán
├── docs/         # Tài liệu học tập (local only, không push lên GitHub)
├── app.py        # Giao diện Streamlit
└── requirements.txt
```
