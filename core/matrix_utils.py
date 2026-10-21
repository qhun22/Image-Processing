import numpy as np


def get_window_zero_padded(image: np.ndarray, x: int, y: int, window_size: int) -> np.ndarray:
    """
    Lấy cửa sổ window_size x window_size có tâm tại pixel (x, y) của image,
    dùng zero-padding cho các vị trí nằm ngoài ảnh.

    Quy ước xác nhận từ tài liệu (BÀI 4.pdf, Slide Bài 4 trang 26/33/40):
    cửa sổ luôn có tâm tại pixel đang xét, kích thước cố định window_size x window_size,
    điểm ngoài ảnh được coi là giá trị 0.
    """
    half = window_size // 2
    M, N = image.shape
    window = np.zeros((window_size, window_size), dtype=image.dtype)
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            xi, yj = x + i, y + j
            if 0 <= xi < M and 0 <= yj < N:
                window[i + half, j + half] = image[xi, yj]
    return window


def convolve_formula_2_1(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Nhân chập theo công thức 2.1 của BÀI 4.pdf:
        I⊗T(x,y) = sum_{i=0}^{m-1} sum_{j=0}^{n-1} I(x+i, y+j) * T(i,j)

    Không lật kernel. Zero-padding cho vị trí ngoài ảnh. Output cùng kích thước ảnh gốc.
    Đã verify bằng 2 ví dụ số có đáp án (BÀI 4.pdf trang 1-4), khớp 100%.

    Quy ước anchor cho kernel kích thước lẻ (3x3+) CHƯA được xác nhận bằng ví dụ số độc lập
    đầy đủ trong tài liệu (xem docs/audit/corrections_v5.json) — công thức 2.1 vẫn được áp
    dụng nguyên văn làm mặc định tốt nhất hiện có, không tự thay bằng quy tắc anchor tâm của
    toán tử Laplace (vốn thuộc một chương khác, không phải Dạng 7).
    """
    M, N = image.shape
    m, n = kernel.shape
    result = np.zeros((M, N), dtype=float)
    for x in range(M):
        for y in range(N):
            total = 0.0
            for i in range(m):
                for j in range(n):
                    xi, yj = x + i, y + j
                    if 0 <= xi < M and 0 <= yj < N:
                        total += image[xi, yj] * kernel[i, j]
            result[x, y] = total
    return result


def round_half_up(value: float) -> int:
    """
    Làm tròn kiểu round-half-up (0.5 luôn làm tròn lên), KHÔNG dùng round() mặc định
    của Python (banker's rounding / round-half-to-even).

    Tài liệu không có ví dụ số nào rơi đúng vào trường hợp .5 (xem docs/audit/golden_tests_v3.json
    và corrections_v5.json), nên quy tắc làm tròn .5 CHƯA được xác nhận tuyệt đối từ PDF.
    round-half-up được chọn làm mặc định vì đây là quy ước phổ biến nhất trong giáo trình toán
    phổ thông/đại học Việt Nam, nhưng cần lưu ý đây là một giả định có ghi chú, không phải
    bằng chứng trực tiếp từ tài liệu.
    """
    import math

    return math.floor(value + 0.5)
