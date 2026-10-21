import numpy as np


class ValidationError(ValueError):
    """Lỗi input không hợp lệ, kèm thông báo rõ nguyên nhân cho người dùng."""


def validate_matrix(matrix) -> np.ndarray:
    """Kiểm tra ma trận đầu vào: phải là mảng 2 chiều, các hàng cùng độ dài, chỉ chứa số."""
    if matrix is None:
        raise ValidationError("Ma trận đầu vào không được để trống.")

    if isinstance(matrix, np.ndarray):
        arr = matrix
    else:
        if not isinstance(matrix, (list, tuple)) or len(matrix) == 0:
            raise ValidationError("Ma trận đầu vào phải là danh sách các hàng, không được rỗng.")
        row_lengths = {len(row) for row in matrix}
        if len(row_lengths) > 1:
            raise ValidationError(
                f"Số phần tử mỗi hàng không bằng nhau: các độ dài tìm thấy {sorted(row_lengths)}."
            )
        try:
            arr = np.array(matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Ma trận chứa ký tự không hợp lệ (chỉ chấp nhận số nguyên hoặc số thực)."
            ) from exc

    if arr.ndim != 2:
        raise ValidationError("Ma trận đầu vào phải có đúng 2 chiều (hàng x cột).")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValidationError("Ma trận đầu vào không được rỗng.")
    if not np.isfinite(arr).all():
        raise ValidationError("Ma trận chứa giá trị không hợp lệ (NaN hoặc vô cực).")

    return arr


def validate_window_size(window_size) -> int:
    """Kiểm tra kích thước cửa sổ W: phải là số nguyên dương lẻ."""
    try:
        w = int(window_size)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Kích thước cửa sổ W phải là số nguyên.") from exc
    if w <= 0:
        raise ValidationError("Kích thước cửa sổ W phải là số dương.")
    if w % 2 == 0:
        raise ValidationError(
            f"Kích thước cửa sổ W={w} không hợp lệ: tài liệu chỉ xác nhận cửa sổ kích thước lẻ (vd 3x3)."
        )
    return w


def validate_threshold(theta, allow_negative: bool = False) -> float:
    """Kiểm tra ngưỡng θ."""
    try:
        t = float(theta)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Ngưỡng θ phải là số.") from exc
    if not allow_negative and t < 0:
        raise ValidationError("Ngưỡng θ phải là số không âm (θ ≥ 0) theo định nghĩa trong tài liệu.")
    return t


def validate_new_level(new_level) -> int:
    """Kiểm tra new_level cho cân bằng histogram: số nguyên dương."""
    try:
        n = int(new_level)
    except (TypeError, ValueError) as exc:
        raise ValidationError("new_level phải là số nguyên.") from exc
    if n <= 0:
        raise ValidationError("new_level phải là số nguyên dương (new_level > 0).")
    return n


def validate_g(g) -> int:
    """Kiểm tra G cho tách ngưỡng tự động: số nguyên dương."""
    try:
        g_val = int(g)
    except (TypeError, ValueError) as exc:
        raise ValidationError("G phải là số nguyên.") from exc
    if g_val <= 0:
        raise ValidationError("G phải là số nguyên dương (G > 0).")
    return g_val


def validate_k(k, window_size: int) -> int:
    """Kiểm tra k cho lọc trung bình theo k giá trị gần nhất: 1 <= k <= W*W."""
    try:
        k_val = int(k)
    except (TypeError, ValueError) as exc:
        raise ValidationError("k phải là số nguyên.") from exc
    max_k = window_size * window_size
    if k_val < 1 or k_val > max_k:
        raise ValidationError(
            f"k={k_val} không hợp lệ: phải trong khoảng [1, {max_k}] (kích thước cửa sổ {window_size}x{window_size})."
        )
    return k_val


def validate_kernel(kernel) -> np.ndarray:
    """Kiểm tra mẫu T cho nhân chập: phải là ma trận 2 chiều không rỗng."""
    if kernel is None:
        raise ValidationError("Mẫu T không được để trống.")
    try:
        arr = validate_matrix(kernel)
    except ValidationError as exc:
        raise ValidationError(f"Mẫu T không hợp lệ: {exc}") from exc
    return arr
