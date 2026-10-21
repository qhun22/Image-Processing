import numpy as np

from core.validators import validate_matrix, validate_window_size, validate_threshold
from core.matrix_utils import get_window_zero_padded


def median_filter(image, window_size, threshold) -> dict:
    """
    Lọc trung vị có điều kiện ngưỡng (BÀI 4.pdf trang 6-7; Slide Bài 4 trang 23-29):
        TV(P) = trung vị của cửa sổ W(P)
        I'(P) = I(P)   nếu |I(P) - TV(P)| <= theta
                 TV(P)  nếu ngược lại

    Cửa sổ W có tâm tại pixel đang xét, zero-padding cho vị trí ngoài ảnh (đã verify bằng
    reverse-engineering khớp 100% với ví dụ số trong tài liệu).
    """
    img = validate_matrix(image)
    w = validate_window_size(window_size)
    theta = validate_threshold(threshold)

    M, N = img.shape
    median_matrix = np.zeros((M, N))
    abs_diff = np.zeros((M, N))
    result = np.zeros((M, N))
    window_logs = []

    for x in range(M):
        for y in range(N):
            window = get_window_zero_padded(img, x, y, w)
            tv = float(np.median(window))
            diff = abs(img[x, y] - tv)
            median_matrix[x, y] = tv
            abs_diff[x, y] = diff
            result[x, y] = img[x, y] if diff <= theta else tv
            window_logs.append({
                "position": (x, y),
                "center_value": float(img[x, y]),
                "window": window.tolist(),
                "median": tv,
                "abs_diff": diff,
                "replaced": diff > theta,
            })

    return {
        "result": result,
        "median_matrix": median_matrix,
        "abs_diff_matrix": abs_diff,
        "window_logs": window_logs,
        "steps": _build_steps(w, theta, window_logs),
    }


def _build_steps(w, theta, window_logs) -> list:
    steps = [f"Cửa sổ W={w}x{w}, ngưỡng theta={theta}, zero-padding ngoài ảnh."]
    for log in window_logs:
        x, y = log["position"]
        action = "thay bằng TV" if log["replaced"] else "giữ nguyên"
        steps.append(
            f"Tại ({x},{y}): I={log['center_value']}, TV={log['median']}, "
            f"|I-TV|={log['abs_diff']} {'>' if log['replaced'] else '<='} theta -> {action}"
        )
    return steps
