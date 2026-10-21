import numpy as np

from core.validators import validate_matrix, validate_window_size, validate_threshold
from core.matrix_utils import get_window_zero_padded, round_half_up


def mean_filter(image, window_size, threshold) -> dict:
    """
    Lọc trung bình có điều kiện ngưỡng (BÀI 4.pdf trang 8-10; Slide Bài 4 trang 30-37):
        TB(P) = trung bình cộng cửa sổ W(P)
        I'(P) = I(P)   nếu |I(P) - TB(P)| <= theta
                 TB(P)  nếu ngược lại

    Cửa sổ W có tâm tại pixel đang xét, zero-padding, LUÔN chia cho diện tích cửa sổ cố định
    (window_size^2), không chia theo số điểm hợp lệ (đã verify bằng reverse-engineering khớp
    100% với tài liệu).

    Làm tròn: round-to-nearest, xác nhận trực tiếp bằng ví dụ "3.89 lấy tròn làm 4"
    (Slide Bài 4 trang 33) — ĐÃ GIẢI QUYẾT dứt điểm mâu thuẫn 3 vs 4 trong corrections_v5.json,
    giá trị đúng là 4.
    """
    img = validate_matrix(image)
    w = validate_window_size(window_size)
    theta = validate_threshold(threshold)

    M, N = img.shape
    area = w * w
    mean_matrix = np.zeros((M, N))
    abs_diff = np.zeros((M, N))
    result = np.zeros((M, N))
    window_logs = []

    for x in range(M):
        for y in range(N):
            window = get_window_zero_padded(img, x, y, w)
            raw_mean = float(np.sum(window)) / area
            tb = round_half_up(raw_mean)
            diff = abs(img[x, y] - tb)
            mean_matrix[x, y] = tb
            abs_diff[x, y] = diff
            result[x, y] = img[x, y] if diff <= theta else tb
            window_logs.append({
                "position": (x, y),
                "center_value": float(img[x, y]),
                "window": window.tolist(),
                "raw_mean": raw_mean,
                "mean_rounded": tb,
                "abs_diff": diff,
                "replaced": diff > theta,
            })

    return {
        "result": result,
        "mean_matrix": mean_matrix,
        "abs_diff_matrix": abs_diff,
        "window_logs": window_logs,
        "steps": _build_steps(w, theta, window_logs),
    }


def _build_steps(w, theta, window_logs) -> list:
    steps = [f"Cửa sổ W={w}x{w}, ngưỡng theta={theta}, zero-padding, chia cố định cho {w*w}."]
    for log in window_logs:
        x, y = log["position"]
        action = "thay bằng TB" if log["replaced"] else "giữ nguyên"
        steps.append(
            f"Tại ({x},{y}): I={log['center_value']}, TB={log['raw_mean']:.4f} lấy tròn "
            f"= {log['mean_rounded']}, |I-TB|={log['abs_diff']} "
            f"{'>' if log['replaced'] else '<='} theta -> {action}"
        )
    return steps
