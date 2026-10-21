import numpy as np

from core.validators import validate_matrix, validate_window_size, validate_threshold, validate_k
from core.matrix_utils import get_window_zero_padded, round_half_up


def k_nearest_mean_filter(image, window_size, threshold, k) -> dict:
    """
    Lọc trung bình theo k giá trị gần nhất (BÀI 4.pdf trang 10-11; Slide Bài 4 trang 38-44):
        B1: tìm k giá trị trong cửa sổ W(P) gần I(P) nhất (theo |I(q)-I(P)|, cường độ,
            KHÔNG theo khoảng cách không gian)
        B2: TBk(P) = trung bình k giá trị đó
        B3: I'(P) = I(P) nếu |I(P)-TBk(P)| <= theta, ngược lại = TBk(P)

    Cửa sổ W có tâm tại pixel đang xét, zero-padding (đã verify khớp 100%).

    QUY TẮC TIE-BREAKING CHƯA XÁC ĐỊNH TỪ TÀI LIỆU: khi có từ 2 giá trị KHÁC NHAU trở lên
    cùng khoảng cách tới I(P) và việc chọn giữa chúng ảnh hưởng đến kết quả cuối (tức là
    không phải tất cả các giá trị đồng hạng đều giống hệt nhau), hàm này sẽ đánh dấu
    tie_break_used=True và áp dụng quy tắc tạm thời "ưu tiên giá trị xuất hiện trước theo
    thứ tự quét cửa sổ (trái-phải, trên-xuống)" — đây là quy tắc CHƯA CÓ CĂN CỨ TỪ TÀI LIỆU,
    chỉ để chương trình chạy được, và PHẢI được hiển thị rõ ràng cho người dùng, không được
    báo cáo như một đáp án chắc chắn duy nhất.
    """
    img = validate_matrix(image)
    w = validate_window_size(window_size)
    theta = validate_threshold(threshold)
    k_val = validate_k(k, w)

    M, N = img.shape
    result = np.zeros((M, N))
    tbk_matrix = np.zeros((M, N))
    abs_diff = np.zeros((M, N))
    window_logs = []
    any_tie_break_used = False

    for x in range(M):
        for y in range(N):
            window = get_window_zero_padded(img, x, y, w)
            center = img[x, y]
            flat = window.flatten()
            # Sắp xếp theo khoảng cách; dùng sắp xếp ổn định (stable sort) để bảo toàn
            # thứ tự quét ban đầu khi có đồng hạng (quy tắc tạm thời, xem docstring).
            order = sorted(range(len(flat)), key=lambda idx: abs(flat[idx] - center))
            k_nearest = [flat[idx] for idx in order[:k_val]]

            tie_break_used = False
            if k_val < len(flat):
                boundary_distance = abs(flat[order[k_val - 1]] - center)
                next_distance = abs(flat[order[k_val]] - center)
                if boundary_distance == next_distance:
                    tied_values = {flat[order[i]] for i in range(len(flat)) if abs(flat[order[i]] - center) == boundary_distance}
                    if len(tied_values) > 1:
                        tie_break_used = True
                        any_tie_break_used = True

            tb_k_raw = sum(k_nearest) / k_val
            tb_k = round_half_up(tb_k_raw)
            diff = abs(center - tb_k)

            tbk_matrix[x, y] = tb_k
            abs_diff[x, y] = diff
            result[x, y] = center if diff <= theta else tb_k

            window_logs.append({
                "position": (x, y),
                "center_value": float(center),
                "window": window.tolist(),
                "k_nearest_values": k_nearest,
                "tbk_raw": tb_k_raw,
                "tbk_rounded": tb_k,
                "abs_diff": diff,
                "replaced": diff > theta,
                "tie_break_used": tie_break_used,
            })

    return {
        "result": result,
        "tbk_matrix": tbk_matrix,
        "abs_diff_matrix": abs_diff,
        "window_logs": window_logs,
        "tie_break_used_anywhere": any_tie_break_used,
        "steps": _build_steps(w, theta, k_val, window_logs, any_tie_break_used),
    }


def _build_steps(w, theta, k_val, window_logs, any_tie_break_used) -> list:
    steps = [f"Cửa sổ W={w}x{w}, ngưỡng theta={theta}, k={k_val}, zero-padding."]
    if any_tie_break_used:
        steps.append(
            "CẢNH BÁO: có ít nhất một pixel gặp trường hợp hòa khoảng cách giữa các giá trị "
            "KHÁC NHAU khi chọn k giá trị gần nhất. Tài liệu không xác định quy tắc xử lý "
            "trường hợp này — chương trình đã tạm dùng thứ tự quét cửa sổ để chọn, đây KHÔNG "
            "phải một đáp án chắc chắn duy nhất."
        )
    for log in window_logs:
        x, y = log["position"]
        action = "thay bằng TBk" if log["replaced"] else "giữ nguyên"
        tie_note = " [CÓ HÒA KHOẢNG CÁCH - chưa xác định]" if log["tie_break_used"] else ""
        steps.append(
            f"Tại ({x},{y}): I={log['center_value']}, {k_val} giá trị gần nhất={log['k_nearest_values']}, "
            f"TBk={log['tbk_raw']:.4f} lấy tròn={log['tbk_rounded']}, |I-TBk|={log['abs_diff']} "
            f"{'>' if log['replaced'] else '<='} theta -> {action}{tie_note}"
        )
    return steps
