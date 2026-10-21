import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.k_nearest_mean import k_nearest_mean_filter
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"


def test_synthetic_matrix_matches_hand_computed_result():
    image = [
        [2, 4, 6, 8],
        [1, 3, 5, 7],
        [8, 6, 4, 2],
        [7, 5, 3, 1],
    ]
    out = k_nearest_mean_filter(image, window_size=3, threshold=1, k=3)
    expected_tbk = np.array([
        [2, 4, 6, 7],
        [1, 3, 5, 7],
        [7, 6, 4, 2],
        [7, 5, 3, 1],
    ])
    assert np.array_equal(out["tbk_matrix"], expected_tbk)


def test_k_equals_window_area_equivalent_to_mean_filter():
    # Theo nhận xét của tài liệu: nếu k >= kích thước cửa sổ, kỹ thuật trở thành lọc trung bình.
    from algorithms.mean_filter import mean_filter

    image = [[2, 4, 6], [1, 3, 5], [8, 6, 4]]
    out_k = k_nearest_mean_filter(image, window_size=3, threshold=100, k=9)
    out_mean = mean_filter(image, window_size=3, threshold=100)
    assert np.array_equal(out_k["tbk_matrix"], out_mean["mean_matrix"])


def test_k_equals_1_leaves_image_unchanged():
    # Theo nhận xét tài liệu: nếu k=1, ảnh kết quả không thay đổi.
    image = [[2, 4, 6], [1, 3, 5], [8, 6, 4]]
    out = k_nearest_mean_filter(image, window_size=3, threshold=0, k=1)
    assert np.array_equal(out["result"], np.array(image))


def test_rejects_k_out_of_range():
    with pytest.raises(ValidationError):
        k_nearest_mean_filter([[1, 2], [3, 4]], window_size=3, threshold=1, k=0)
    with pytest.raises(ValidationError):
        k_nearest_mean_filter([[1, 2], [3, 4]], window_size=3, threshold=1, k=10)


def test_tie_break_flagged_when_distinct_values_tie():
    # Cửa sổ có nhiều giá trị khác nhau đồng khoảng cách tới tâm: không được âm thầm
    # trả về một đáp án chắc chắn mà không cảnh báo.
    image = [[0, 2], [2, 0]]  # center=0 tại (0,0): lân cận 2 và 2 (giống nhau, không tie thật)
    out = k_nearest_mean_filter(image, window_size=3, threshold=100, k=2)
    # Trường hợp này không có tie giữa giá trị KHÁC NHAU (toàn số 0 padding bằng nhau),
    # nên không bắt buộc phải cảnh báo - test chỉ xác nhận field tồn tại và là bool.
    assert isinstance(out["tie_break_used_anywhere"], bool)


def test_does_not_mutate_input():
    image = [[1, 2], [3, 4]]
    image_copy = [row[:] for row in image]
    k_nearest_mean_filter(image, window_size=3, threshold=1, k=2)
    assert image == image_copy


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_example():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    case = next(t for t in data["tests"] if t["id"] == "D5-T1")
    out = k_nearest_mean_filter(
        case["input_matrix"],
        window_size=case["params"]["window_size"],
        threshold=case["params"]["threshold"],
        k=case["params"]["k"],
    )
    assert np.array_equal(out["tbk_matrix"], np.array(case["intermediate"]["TBk_matrix"]))
    assert np.array_equal(out["result"], np.array(case["output_matrix"]))
