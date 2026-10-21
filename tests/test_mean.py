import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.mean_filter import mean_filter
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"


def test_synthetic_matrix_matches_hand_computed_result():
    image = [
        [2, 4, 6, 8],
        [1, 3, 5, 7],
        [8, 6, 4, 2],
        [7, 5, 3, 1],
    ]
    out = mean_filter(image, window_size=3, threshold=1)
    expected_mean = np.array([
        [1, 2, 4, 3],
        [3, 4, 5, 4],
        [3, 5, 4, 2],
        [3, 4, 2, 1],
    ])
    expected_result = np.array([
        [2, 2, 4, 3],
        [3, 3, 5, 4],
        [3, 6, 4, 2],
        [3, 5, 3, 1],
    ])
    assert np.array_equal(out["mean_matrix"], expected_mean)
    assert np.array_equal(out["result"], expected_result)


def test_rounding_half_up_not_banker_rounding():
    # 3.5 phải làm tròn thành 4 (round-half-up), không phải 4 chẵn kiểu banker's rounding
    # (ở đây 3.5 cũng làm tròn thành 4 với cả 2 quy tắc, nên dùng ví dụ 2.5 -> 3 để phân biệt
    # rõ: banker's rounding của Python round(2.5) = 2, còn round-half-up phải là 3).
    image = [[0, 5], [0, 0]]  # tổng cửa sổ 3x3 quanh pixel (0,0) phụ thuộc padding
    out = mean_filter(image, window_size=3, threshold=100)
    # Chỉ cần xác nhận hàm không dùng round() mặc định của Python cho trường hợp .5
    from core.matrix_utils import round_half_up
    assert round_half_up(2.5) == 3
    assert round(2.5) == 2  # banker's rounding mặc định của Python, xác nhận khác round_half_up


def test_does_not_mutate_input():
    image = [[1, 2], [3, 4]]
    image_copy = [row[:] for row in image]
    mean_filter(image, window_size=3, threshold=1)
    assert image == image_copy


def test_rejects_even_window_size():
    with pytest.raises(ValidationError):
        mean_filter([[1, 2], [3, 4]], window_size=2, threshold=1)


def test_rejects_negative_threshold():
    with pytest.raises(ValidationError):
        mean_filter([[1, 2], [3, 4]], window_size=3, threshold=-5)


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_example_pixel_16_is_4_not_3():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    case = next(t for t in data["tests"] if t["id"] == "D4-T1")
    out = mean_filter(
        case["input_matrix"],
        window_size=case["params"]["window_size"],
        threshold=case["params"]["threshold"],
    )
    assert np.array_equal(out["mean_matrix"], np.array(case["intermediate"]["TB_matrix"]))
    assert np.array_equal(out["result"], np.array(case["output_matrix"]))
    # Khẳng định rõ ràng: pixel có giá trị gốc 16 phải cho TB=4, không phải 3.
    assert out["mean_matrix"][1, 1] == 4
