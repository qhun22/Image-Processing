import numpy as np

from core.validators import validate_matrix, validate_new_level
from core.matrix_utils import round_half_up


def equalize_histogram(image, new_level) -> dict:
    """
    Cân bằng histogram theo công thức (Bài giảng 2.pdf trang 6-7; Slide Bài 2 trang 28-29):
        TB = M*N / new_level
        f(g) = max(0, round(t(g)/TB - 1))
    với t(g) là tổng tích lũy số điểm ảnh có mức xám <= g.

    Đã verify bằng 2 ví dụ số có đáp án đầy đủ (Slide Bài 2 trang 29 và 30), khớp 100%.
    """
    img = validate_matrix(image)
    level = validate_new_level(new_level)

    M, N = img.shape
    total_pixels = M * N
    TB = total_pixels / level

    int_image = np.rint(img).astype(int)
    max_gray = int(int_image.max())

    histogram = {g: int(np.sum(int_image == g)) for g in range(max_gray + 1)}

    cumulative = {}
    running = 0
    for g in range(max_gray + 1):
        running += histogram[g]
        cumulative[g] = running

    mapping = {}
    for g in range(max_gray + 1):
        raw = cumulative[g] / TB - 1
        mapping[g] = max(0, round_half_up(raw))

    result = np.vectorize(lambda g: mapping[int(g)])(int_image)

    return {
        "result": result,
        "histogram": histogram,
        "cumulative": cumulative,
        "TB": TB,
        "mapping": mapping,
        "steps": _build_steps(M, N, level, TB, histogram, cumulative, mapping),
    }


def _build_steps(M, N, level, TB, histogram, cumulative, mapping) -> list:
    steps = [
        f"Kích thước ảnh: M*N = {M}*{N} = {M * N} điểm ảnh.",
        f"TB = M*N/new_level = {M * N}/{level} = {TB:.4f}",
    ]
    for g in sorted(histogram):
        h = histogram[g]
        t = cumulative[g]
        f = mapping[g]
        steps.append(
            f"g={g}: h(g)={h}, t(g)={t}, f(g)=max(0, round({t}/{TB:.4f} - 1)) = {f}"
        )
    return steps
