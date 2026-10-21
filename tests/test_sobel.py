import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.sobel import sobel_edge_detection
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"


def test_synthetic_matrix_matches_hand_computed_result():
    image = [
        [3, 2, 1, 0],
        [6, 5, 4, 3],
        [9, 8, 7, 6],
        [2, 1, 0, 9],
    ]
    out = sobel_edge_detection(image, threshold=10)
    expected_mag = np.array([
        [26, 26, 22, 16],
        [38, 32, 32, 34],
        [34, 24, 8, 26],
        [36, 38, 42, 26],
    ])
    expected_result = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ])
    assert np.array_equal(out["magnitude"], expected_mag)
    assert np.array_equal(out["result"], expected_result)


def test_uses_sum_of_absolutes_not_euclidean_norm():
    # Xác nhận công thức dùng |Gx|+|Gy|, không phải sqrt(Gx^2+Gy^2)
    image = [[0, 0, 0], [0, 10, 0], [0, 0, 0]]
    out = sobel_edge_detection(image, threshold=0)
    manual_sum = np.abs(out["Gx"]) + np.abs(out["Gy"])
    assert np.array_equal(out["magnitude"], manual_sum)


def test_does_not_mutate_input():
    image = [[1, 2], [3, 4]]
    image_copy = [row[:] for row in image]
    sobel_edge_detection(image, threshold=5)
    assert image == image_copy


def test_rejects_negative_threshold():
    with pytest.raises(ValidationError):
        sobel_edge_detection([[1, 2], [3, 4]], threshold=-1)


def test_source_data_dispute_note_present():
    image = [[1, 2], [3, 4]]
    out = sobel_edge_detection(image, threshold=5)
    assert "source_data_note" in out
    assert "22" in out["source_data_note"] and "26" in out["source_data_note"]


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_example_with_documented_discrepancy():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    case = next(t for t in data["tests"] if t["id"] == "D6-T1")
    out = sobel_edge_detection(case["input_matrix"], threshold=case["params"]["threshold"])

    doc_magnitude = np.array(case["intermediate"]["magnitude_doc_table"])
    computed_magnitude = out["magnitude"]

    # Biết trước có đúng 1 ô sai lệch (lỗi cộng trong tài liệu gốc, đã xác nhận bằng mắt).
    diff_mask = computed_magnitude != doc_magnitude
    assert diff_mask.sum() == 1, "Số ô sai lệch với tài liệu khác với dự kiến (chỉ 1 ô đã biết)."
    mismatched_pos = np.argwhere(diff_mask)[0]
    assert tuple(mismatched_pos) == (5, 6), "Vị trí sai lệch khác với vị trí đã xác nhận trước đó."

    # Kết quả nhị phân cuối cùng vẫn phải khớp vì cả 22 và 26 đều < ngưỡng 34.
    assert np.array_equal(out["result"], np.array(case["output_matrix"]))
