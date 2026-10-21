"""
Kiểm thử các thuật toán với ma trận lớn hơn (6x7, 8x9, 12x13) và cửa sổ lớn (5x5, 7x7),
cùng mẫu T nhiều kích thước cho Dạng 7. Mỗi test kiểm tra một thuộc tính CỤ THỂ có thể
phát hiện lỗi triển khai thực sự (shape đúng, biên không vỡ, tính chất toán học được giữ
nguyên ở quy mô lớn hơn), KHÔNG chỉ chạy cho có số lượng.

Ma trận tổng hợp dùng công thức xác định (không ngẫu nhiên) để tái lập được, không sao
chép từ giáo trình.
"""
import numpy as np
import pytest

from algorithms.histogram import equalize_histogram
from algorithms.auto_threshold import auto_threshold
from algorithms.median_filter import median_filter
from algorithms.mean_filter import mean_filter
from algorithms.k_nearest_mean import k_nearest_mean_filter
from algorithms.sobel import sobel_edge_detection
from algorithms.convolution import convolve


def make_matrix(rows, cols, modulo=17, seed_offset=0):
    """Ma trận tổng hợp xác định, giá trị trong [0, modulo-1]."""
    return [[(i * cols + j + seed_offset) % modulo for j in range(cols)] for i in range(rows)]


SHAPES = [(6, 7), (8, 9), (12, 13)]
WINDOW_SIZES = [3, 5, 7]


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_histogram_output_shape_matches_input_at_scale(rows, cols):
    image = make_matrix(rows, cols)
    out = equalize_histogram(image, new_level=8)
    assert out["result"].shape == (rows, cols)
    # Mọi giá trị ánh xạ phải nằm trong [0, new_level-1]
    assert out["result"].min() >= 0
    assert out["result"].max() <= 7


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_auto_threshold_binary_result_shape_and_range_at_scale(rows, cols):
    image = make_matrix(rows, cols)
    max_val = max(max(row) for row in image)
    out = auto_threshold(image, G=max_val + 1)
    assert out["unconfirmed_binary_result"].shape == (rows, cols)
    assert set(np.unique(out["unconfirmed_binary_result"]).tolist()).issubset({0, 1})


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_median_filter_shape_preserved_and_no_oob_at_scale(rows, cols, window_size):
    image = make_matrix(rows, cols)
    out = median_filter(image, window_size=window_size, threshold=2)
    assert out["result"].shape == (rows, cols)
    assert out["median_matrix"].shape == (rows, cols)
    # Median của zero-padded window không bao giờ âm vì ảnh gốc và padding đều >= 0
    assert (out["median_matrix"] >= 0).all()


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_mean_filter_shape_preserved_and_result_within_plausible_range(rows, cols, window_size):
    image = make_matrix(rows, cols)
    out = mean_filter(image, window_size=window_size, threshold=2)
    assert out["result"].shape == (rows, cols)
    # Trung bình cửa sổ zero-padded của giá trị không âm luôn không âm, và không vượt
    # quá giá trị lớn nhất có thể trong ảnh gốc (vì padding chỉ thêm số 0, kéo trung bình
    # xuống chứ không bao giờ lên).
    max_in_image = max(max(row) for row in image)
    assert (out["mean_matrix"] >= 0).all()
    assert (out["mean_matrix"] <= max_in_image).all()


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_k_nearest_mean_shape_preserved_at_scale_with_large_window(rows, cols):
    image = make_matrix(rows, cols)
    out = k_nearest_mean_filter(image, window_size=5, threshold=2, k=5)
    assert out["result"].shape == (rows, cols)
    assert out["tbk_matrix"].shape == (rows, cols)
    assert isinstance(out["tie_break_used_anywhere"], bool)


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_sobel_shape_preserved_at_scale(rows, cols):
    image = make_matrix(rows, cols)
    out = sobel_edge_detection(image, threshold=10)
    assert out["Gx"].shape == (rows, cols)
    assert out["Gy"].shape == (rows, cols)
    assert out["result"].shape == (rows, cols)
    assert set(np.unique(out["result"]).tolist()).issubset({0, 1})


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
def test_convolution_output_shape_matches_input_for_various_kernel_sizes(rows, cols, kernel_size):
    image = make_matrix(rows, cols)
    kernel = make_matrix(kernel_size, kernel_size, modulo=3, seed_offset=1)
    out = convolve(image, kernel)
    assert out["result"].shape == (rows, cols), (
        f"Output phải giữ nguyên kích thước ảnh gốc ({rows}x{cols}) theo công thức 2.1, "
        f"không phụ thuộc kích thước kernel {kernel_size}x{kernel_size}."
    )


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_convolution_odd_kernel_always_flags_unconfirmed_anchor_at_any_size(kernel_size):
    # Mọi kernel lẻ kích thước >= 3 đều phải bị đánh dấu anchor chưa xác nhận, không chỉ 3x3.
    image = make_matrix(10, 10)
    kernel = make_matrix(kernel_size, kernel_size, modulo=3)
    out = convolve(image, kernel)
    assert out["anchor_rule_confirmed"] is False


@pytest.mark.parametrize("kernel_size", [2, 4, 6])
def test_convolution_even_kernel_is_confirmed_at_any_size(kernel_size):
    # Mọi kernel chẵn đều dùng công thức 2.1 nguyên văn đã verify, không có tâm mơ hồ.
    image = make_matrix(10, 10)
    kernel = make_matrix(kernel_size, kernel_size, modulo=3)
    out = convolve(image, kernel)
    assert out["anchor_rule_confirmed"] is True


def test_median_and_mean_agree_on_uniform_matrix_at_scale():
    # Trên ma trận đồng nhất (mọi giá trị bằng nhau), median và mean phải cho CÙNG kết quả
    # tại mọi pixel trong (không bị ảnh hưởng bởi padding ở biên khi window nhỏ hơn ảnh)
    # -- một bất biến toán học đơn giản giúp phát hiện lỗi tính median/mean bị lệch.
    rows, cols = 12, 13
    value = 7
    image = [[value] * cols for _ in range(rows)]
    out_median = median_filter(image, window_size=3, threshold=0)
    out_mean = mean_filter(image, window_size=3, threshold=0)
    # Tại các pixel đủ xa biên (không bị ảnh hưởng zero-padding), median == mean == value.
    interior = slice(2, -2)
    assert (out_median["median_matrix"][interior, interior] == value).all()
    assert (out_mean["mean_matrix"][interior, interior] == value).all()
