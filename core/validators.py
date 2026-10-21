import math

import numpy as np

MAX_GRAY_LEVEL = 255


class ValidationError(ValueError):
    """Lỗi input không hợp lệ, kèm thông báo rõ nguyên nhân cho người dùng."""


def _to_strict_int(value, field_name: str) -> int:
    """
    Chuyển value sang int, TỪ CHỐI số thực có phần thập phân khác 0 (vd 3.8).

    Không dùng int(value) trực tiếp vì int(3.8) == 3 âm thầm cắt bỏ phần thập phân
    thay vì báo lỗi — đây là hành vi nguy hiểm cho input API (JSON số 3.8 sẽ bị chấp
    nhận và cắt thành 3 mà người dùng không biết).
    """
    if isinstance(value, bool):
        # bool là subclass của int trong Python; loại trừ tường minh để tránh True/False
        # bị hiểu nhầm thành 1/0.
        raise ValidationError(f"{field_name} phải là số nguyên, không phải boolean.")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} không hợp lệ (NaN hoặc vô cực).")
        if value != math.floor(value):
            raise ValidationError(
                f"{field_name}={value} không phải số nguyên (có phần thập phân khác 0)."
            )
        return int(value)

    if isinstance(value, str):
        text = value.strip()
        try:
            as_float = float(text)
        except ValueError as exc:
            raise ValidationError(f"{field_name} phải là số nguyên hợp lệ.") from exc
        return _to_strict_int(as_float, field_name)

    raise ValidationError(f"{field_name} phải là số nguyên.")


def validate_matrix(matrix, check_gray_range: bool = True) -> np.ndarray:
    """
    Kiểm tra ma trận đầu vào: phải là mảng 2 chiều, các hàng cùng độ dài, chỉ chứa số
    hữu hạn (không NaN/Infinity).

    check_gray_range=True (mặc định, dùng cho ảnh I): giá trị phải nằm trong miền mức
    xám hợp lệ [0, 255]. check_gray_range=False (dùng cho mẫu/kernel T): bỏ qua kiểm tra
    miền giá trị, vì mặt nạ như Sobel/Laplace có hệ số âm (vd -1, -2, -4) một cách hợp lệ.
    """
    if matrix is None:
        raise ValidationError("Ma trận đầu vào không được để trống.")

    if isinstance(matrix, np.ndarray):
        arr = matrix.astype(float, copy=True)
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
    if check_gray_range and ((arr < 0).any() or (arr > MAX_GRAY_LEVEL).any()):
        bad_min, bad_max = float(arr.min()), float(arr.max())
        raise ValidationError(
            f"Ma trận chứa giá trị ngoài miền mức xám hợp lệ [0, {MAX_GRAY_LEVEL}] "
            f"(tìm thấy khoảng [{bad_min}, {bad_max}])."
        )

    return arr


def validate_window_size(window_size) -> int:
    """Kiểm tra kích thước cửa sổ W: phải là số nguyên dương lẻ (không chấp nhận 3.8)."""
    w = _to_strict_int(window_size, "Kích thước cửa sổ W")
    if w <= 0:
        raise ValidationError("Kích thước cửa sổ W phải là số dương.")
    if w % 2 == 0:
        raise ValidationError(
            f"Kích thước cửa sổ W={w} không hợp lệ: tài liệu chỉ xác nhận cửa sổ kích thước lẻ (vd 3x3)."
        )
    return w


def validate_threshold(theta, allow_negative: bool = False) -> float:
    """Kiểm tra ngưỡng θ: số hữu hạn, không NaN/Infinity."""
    try:
        t = float(theta)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Ngưỡng θ phải là số.") from exc
    if not math.isfinite(t):
        raise ValidationError("Ngưỡng θ không hợp lệ (NaN hoặc vô cực).")
    if not allow_negative and t < 0:
        raise ValidationError("Ngưỡng θ phải là số không âm (θ ≥ 0) theo định nghĩa trong tài liệu.")
    return t


def validate_new_level(new_level) -> int:
    """Kiểm tra new_level cho cân bằng histogram: số nguyên dương (không chấp nhận 3.8)."""
    n = _to_strict_int(new_level, "new_level")
    if n <= 0:
        raise ValidationError("new_level phải là số nguyên dương (new_level > 0).")
    return n


def validate_g(g) -> int:
    """Kiểm tra G cho tách ngưỡng tự động: số nguyên dương (không chấp nhận 3.8)."""
    g_val = _to_strict_int(g, "G")
    if g_val <= 0:
        raise ValidationError("G phải là số nguyên dương (G > 0).")
    return g_val


def validate_p(p, g: int) -> int:
    """
    Kiểm tra P cho tách ngưỡng tự động: số nguyên dương, và PHẢI nhất quán với G
    (P phải đủ lớn để chứa toàn bộ số điểm ảnh có mức xám trong [0, G-1]; việc kiểm tra
    P so với tổng thực tế của ảnh nằm ở algorithms/auto_threshold.py vì cần dữ liệu ảnh).
    """
    p_val = _to_strict_int(p, "P")
    if p_val <= 0:
        raise ValidationError("P phải là số nguyên dương (P > 0).")
    return p_val


def validate_k(k, window_size: int) -> int:
    """Kiểm tra k cho lọc trung bình theo k giá trị gần nhất: 1 <= k <= W*W (không chấp nhận 2.5)."""
    k_val = _to_strict_int(k, "k")
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
    if isinstance(kernel, (list, tuple)) and len(kernel) == 0:
        raise ValidationError("Mẫu T không hợp lệ: mẫu T không được rỗng.")
    try:
        arr = validate_matrix(kernel, check_gray_range=False)
    except ValidationError as exc:
        raise ValidationError(f"Mẫu T không hợp lệ: {exc}") from exc
    return arr
