import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.median_filter import median_filter
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"


def test_synthetic_matrix_matches_hand_computed_result():
    image = [
        [2, 4, 6, 8],
        [1, 3, 5, 7],
        [8, 6, 4, 2],
        [7, 5, 3, 1],
    ]
    out = median_filter(image, window_size=3, threshold=1)
    expected_median = np.array([
        [0, 2, 4, 0],
        [2, 4, 5, 4],
        [3, 5, 4, 2],
        [0, 4, 2, 0],
    ])
    expected_result = np.array([
        [0, 2, 4, 0],
        [1, 3, 5, 4],
        [3, 6, 4, 2],
        [0, 5, 3, 1],
    ])
    assert np.array_equal(out["median_matrix"], expected_median)
    assert np.array_equal(out["result"], expected_result)


def test_window_larger_than_image_still_works_with_zero_padding():
    image = [[5]]
    out = median_filter(image, window_size=3, threshold=0)
    # cửa sổ 3x3 quanh pixel duy nhất, với 8 giá trị 0 padding + 1 giá trị 5
    assert out["median_matrix"][0, 0] == 0


def test_does_not_mutate_input():
    image = [[1, 2], [3, 4]]
    image_copy = [row[:] for row in image]
    median_filter(image, window_size=3, threshold=1)
    assert image == image_copy


def test_rejects_even_window_size():
    with pytest.raises(ValidationError):
        median_filter([[1, 2], [3, 4]], window_size=4, threshold=1)


def test_rejects_negative_threshold():
    with pytest.raises(ValidationError):
        median_filter([[1, 2], [3, 4]], window_size=3, threshold=-1)


def test_rejects_ragged_matrix():
    with pytest.raises(ValidationError):
        median_filter([[1, 2], [3]], window_size=3, threshold=1)


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_example():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    case = next(t for t in data["tests"] if t["id"] == "D3-T1")
    out = median_filter(
        case["input_matrix"],
        window_size=case["params"]["window_size"],
        threshold=case["params"]["threshold"],
    )
    assert np.array_equal(out["median_matrix"], np.array(case["intermediate"]["TV_matrix"]))
    assert np.array_equal(out["result"], np.array(case["output_matrix"]))
