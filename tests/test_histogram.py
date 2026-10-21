import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.histogram import equalize_histogram
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"


def test_synthetic_matrix_matches_hand_computed_result():
    # Ma trận tổng hợp tự tạo, đáp án tính độc lập bằng tay theo đúng công thức tài liệu.
    image = [
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [2, 2, 3, 3],
        [3, 3, 3, 3],
    ]
    out = equalize_histogram(image, new_level=4)
    expected = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 2],
        [2, 2, 3, 3],
        [3, 3, 3, 3],
    ])
    assert np.array_equal(out["result"], expected)
    assert out["TB"] == 4.0
    assert out["histogram"] == {0: 3, 1: 4, 2: 3, 3: 6}
    assert out["cumulative"] == {0: 3, 1: 7, 2: 10, 3: 16}
    assert out["mapping"] == {0: 0, 1: 1, 2: 2, 3: 3}


def test_single_level_collapses_everything_to_zero():
    image = [[1, 2], [3, 4]]
    out = equalize_histogram(image, new_level=1)
    assert np.all(out["result"] == 0)


def test_does_not_mutate_input():
    image = [[1, 2], [3, 4]]
    image_copy = [row[:] for row in image]
    equalize_histogram(image, new_level=2)
    assert image == image_copy


def test_rejects_ragged_matrix():
    with pytest.raises(ValidationError):
        equalize_histogram([[1, 2], [3]], new_level=2)


def test_rejects_invalid_new_level():
    with pytest.raises(ValidationError):
        equalize_histogram([[1, 2], [3, 4]], new_level=0)
    with pytest.raises(ValidationError):
        equalize_histogram([[1, 2], [3, 4]], new_level=-3)


def test_rejects_non_numeric_matrix():
    with pytest.raises(ValidationError):
        equalize_histogram([[1, "a"], [3, 4]], new_level=2)


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_examples():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    golden_cases = [t for t in data["tests"] if t["id"].startswith("D1") and t["verification_status"] == "VERIFIED"]
    assert len(golden_cases) >= 2

    for case in golden_cases:
        out = equalize_histogram(case["input_matrix"], new_level=case["params"]["new_level"])
        expected_mapping = {int(k): v for k, v in case["intermediate"]["mapping_f_g"].items()}
        # Golden data chỉ liệt kê f(g) cho các mức xám thực sự xuất hiện trong ảnh;
        # out["mapping"] tính cho mọi g từ 0..max nên chỉ so khớp các g có trong golden data.
        for g, expected_f in expected_mapping.items():
            assert out["mapping"][g] == expected_f, f"Mismatch in {case['id']} at g={g}"
        if "output_matrix" in case:
            assert np.array_equal(out["result"], np.array(case["output_matrix"])), f"Output mismatch in {case['id']}"
