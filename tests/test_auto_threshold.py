import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.auto_threshold import auto_threshold
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"
GOLDEN_PATH_V4 = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v4_addendum.json"


def test_synthetic_matrix_matches_hand_computed_result():
    # Ma trận tổng hợp tự tạo (4x4), đáp án tính độc lập bằng công thức tài liệu.
    image = [
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [2, 2, 3, 3],
        [3, 3, 3, 3],
    ]
    out = auto_threshold(image, G=4)
    assert out["threshold"] == 1
    assert out["F_theta"] == pytest.approx(0.08035714285714302, abs=1e-6)
    assert out["A"][3] is None
    assert out["F"][3] is None
    assert 3 in out["skipped_degenerate_classes"]
    assert out["moment"][3] == pytest.approx(1.75)


def test_g_smaller_than_max_gray_level_is_rejected_not_silently_truncated():
    # Ảnh có mức xám tối đa = 5, nhưng G=4 chỉ cho phép xét đến mức 3.
    # Phải báo lỗi rõ ràng, KHÔNG được âm thầm bỏ sót pixel có giá trị 4, 5.
    image = [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(ValueError, match="G=4"):
        auto_threshold(image, G=4)


def test_g_exactly_matching_max_gray_level_is_accepted():
    image = [[0, 1, 2], [3, 4, 5]]
    out = auto_threshold(image, G=6)  # mức xám 0..5 -> G phải >= 6
    assert out["cumulative"][5] == 6  # tổng mọi pixel đã được tính, không bỏ sót


def test_p_inconsistent_with_actual_pixel_count_is_rejected():
    # Ảnh 2x2 = 4 điểm ảnh, nhưng truyền P=100 (không khớp thực tế) phải bị từ chối,
    # không được âm thầm dùng P sai làm lệch kết quả A(g), F(g).
    image = [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match="không nhất quán"):
        auto_threshold(image, G=4, P=100)


def test_p_consistent_with_actual_pixel_count_is_accepted():
    image = [[0, 1], [2, 3]]
    out = auto_threshold(image, G=4, P=4)
    assert out["threshold"] is not None


def test_degenerate_lower_class_empty_is_excluded_from_argmax():
    # g=0 nhưng không có pixel nào mang giá trị 0 -> t(0)=0 (lớp dưới rỗng tại g=0)
    # phải bị loại khỏi candidates, không được tính A(0)=0/something một cách sai lệch.
    image = [[1, 1], [2, 2]]
    out = auto_threshold(image, G=3)
    assert out["cumulative"][0] == 0
    assert 0 in out["skipped_degenerate_classes"]
    assert out["A"][0] is None


def test_division_by_zero_raises_when_all_candidates_degenerate():
    # G=1 nghĩa là chỉ có g=0, và t(0)=P luôn đúng -> mọi ứng viên đều bị loại vì chia cho 0.
    image = [[0, 0], [0, 0]]
    with pytest.raises(ValueError):
        auto_threshold(image, G=1)


def test_does_not_mutate_input():
    image = [[1, 2], [3, 0]]
    image_copy = [row[:] for row in image]
    auto_threshold(image, G=4)
    assert image == image_copy


def test_rejects_invalid_g():
    with pytest.raises(ValidationError):
        auto_threshold([[1, 2], [3, 4]], G=0)
    with pytest.raises(ValidationError):
        auto_threshold([[1, 2], [3, 4]], G=-1)


def test_rejects_non_integer_g():
    with pytest.raises(ValidationError):
        auto_threshold([[1, 2], [3, 4]], G=3.8)


def test_rejects_ragged_matrix():
    with pytest.raises(ValidationError):
        auto_threshold([[1, 2], [3]], G=4)


def test_binary_result_not_exposed_as_confirmed_answer():
    image = [[0, 0], [1, 1]]
    out = auto_threshold(image, G=2)
    assert out["result_rule_confirmed"] is False
    # Field "result" (đáp án chắc chắn) KHÔNG được tồn tại; chỉ có "unconfirmed_binary_result".
    assert "result" not in out
    assert "unconfirmed_binary_result" in out


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_example_1():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    case = next(t for t in data["tests"] if t["id"] == "D2-T1")
    out = auto_threshold(case["input_matrix"], G=case["params"]["G"], P=case["params"]["P"])
    assert out["threshold"] == case["output"]["threshold_theta"]
    assert out["F_theta"] == pytest.approx(case["output"]["F_theta"], abs=0.01)


@pytest.mark.skipif(not GOLDEN_PATH_V4.exists(), reason="golden test data local-only, not present")
def test_against_golden_example_2():
    data = json.loads(GOLDEN_PATH_V4.read_text(encoding="utf-8"))
    case = next(t for t in data["new_tests_found"] if t["id"] == "D2-T2")
    out = auto_threshold(case["input_matrix"], G=case["params"]["G"], P=case["params"]["P"])
    assert out["threshold"] == case["output"]["threshold_theta"]
    assert out["F_theta"] == pytest.approx(case["output"]["F_theta_approx"], abs=0.02)
