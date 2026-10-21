import numpy as np

from core.validators import validate_matrix, validate_threshold

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def sobel_edge_detection(image, threshold) -> dict:
    """
    Phát hiện biên Sobel (Bài 6.pdf trang 5-6; Slide Bài 6 trang 20-26):
        Gx = I ⊗ Sx, Gy = I ⊗ Sy   (anchor tâm mặt nạ, zero-padding — verify khớp 49/49 ô)
        magnitude = |Gx| + |Gy|
        I'(x,y) = 1 nếu magnitude >= theta, ngược lại 0

    Dùng tổng trị tuyệt đối |Gx|+|Gy|, KHÔNG dùng sqrt(Gx^2+Gy^2), theo đúng ví dụ số đã verify.
    """
    img = validate_matrix(image)
    theta = validate_threshold(threshold)

    Gx = _convolve_center_anchor(img, SOBEL_X)
    Gy = _convolve_center_anchor(img, SOBEL_Y)
    magnitude = np.abs(Gx) + np.abs(Gy)
    result = np.where(magnitude >= theta, 1, 0)

    return {
        "Gx": Gx,
        "Gy": Gy,
        "magnitude": magnitude,
        "result": result,
        "steps": _build_steps(theta, Gx, Gy, magnitude, result),
        "source_data_note": (
            "Tại ví dụ 7x7 chuẩn (theta=34) trong tài liệu, ô hàng 6 cột 7 (1-based) có "
            "|Gx|+|Gy| tính đúng công thức = 22, nhưng bảng tổng hợp trong tài liệu gốc ghi "
            "26 (lỗi cộng của chính tài liệu, đã xác nhận bằng kiểm tra hình ảnh PDF gốc, "
            "xem docs/audit/golden_tests_v3.json mục D6-T1). Không ảnh hưởng tới phân loại "
            "nhị phân vì cả 22 và 26 đều nhỏ hơn ngưỡng 34."
        ),
    }


def _convolve_center_anchor(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    M, N = image.shape
    out = np.zeros((M, N))
    half = kernel.shape[0] // 2
    for x in range(M):
        for y in range(N):
            total = 0.0
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    xi, yj = x + i - half, y + j - half
                    if 0 <= xi < M and 0 <= yj < N:
                        total += image[xi, yj] * kernel[i, j]
            out[x, y] = total
    return out


def _build_steps(theta, Gx, Gy, magnitude, result) -> list:
    steps = [
        "Sx = [[-1,0,1],[-2,0,2],[-1,0,1]], Sy = [[1,2,1],[0,0,0],[-1,-2,-1]], anchor tâm, zero-padding.",
        f"Ngưỡng theta={theta}.",
    ]
    M, N = magnitude.shape
    for x in range(M):
        for y in range(N):
            steps.append(
                f"Tại ({x},{y}): Gx={Gx[x,y]:.0f}, Gy={Gy[x,y]:.0f}, "
                f"|Gx|+|Gy|={magnitude[x,y]:.0f} {'>=' if magnitude[x,y]>=theta else '<'} theta "
                f"-> I'={result[x,y]}"
            )
    return steps
