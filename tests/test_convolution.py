import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.convolution import convolve
from core.validators import ValidationError

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "docs" / "audit" / "golden_tests_v3.json"


def test_synthetic_matrix_matches_hand_computed_result():
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 0], [0, 1]]
    out = convolve(image, kernel)
    expected = np.array([
        [6, 8, 3],
        [12, 14, 6],
        [7, 8, 9],
    ])
    assert np.array_equal(out["result"], expected)
    assert out["anchor_rule_confirmed"] is True  # kernel 2x2 chẵn, đã verify


def test_odd_kernel_flags_anchor_unconfirmed():
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]  # Laplace-style 3x3
    out = convolve(image, kernel)
    assert out["anchor_rule_confirmed"] is False
    assert any("CHƯA được xác nhận" in s for s in out["steps"])


def test_does_not_mutate_input():
    image = [[1, 2], [3, 4]]
    kernel = [[1, 1], [0, 1]]
    image_copy = [row[:] for row in image]
    kernel_copy = [row[:] for row in kernel]
    convolve(image, kernel)
    assert image == image_copy
    assert kernel == kernel_copy


def test_rejects_empty_kernel():
    with pytest.raises(ValidationError):
        convolve([[1, 2], [3, 4]], [])


def test_rejects_ragged_kernel():
    with pytest.raises(ValidationError):
        convolve([[1, 2], [3, 4]], [[1, 2], [3]])


def test_does_not_flip_kernel():
    # Ma trận và kernel đều bất đối xứng hoàn toàn, để lật/không lật kernel cho ra
    # kết quả khác nhau rõ ràng TẠI MỌI VỊ TRÍ (không chỉ 1 ô trùng hợp bằng 0).
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 2], [3, 4]]  # bất đối xứng: không có trục đối xứng nào
    out = convolve(image, kernel)

    expected_no_flip = np.array([
        [37, 47, 21],
        [67, 77, 33],
        [23, 26, 9],
    ])
    expected_if_flipped = np.array([
        [23, 33, 24],
        [53, 63, 42],
        [52, 59, 36],
    ])

    assert np.array_equal(out["result"], expected_no_flip)
    assert not np.array_equal(out["result"], expected_if_flipped), (
        "Kết quả trùng với trường hợp kernel bị lật — công thức 2.1 đã bị lật sai."
    )


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden test data local-only, not present")
def test_against_golden_examples():
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = [t for t in data["tests"] if t["id"] in ("D7-T1", "D7-T2")]
    assert len(cases) == 2

    case1 = next(t for t in cases if t["id"] == "D7-T1")
    out1 = convolve(case1["input_matrix"], case1["kernel_T"])
    assert np.array_equal(out1["result"], np.array(case1["output_matrix"]))
    assert out1["anchor_rule_confirmed"] is True

    case2 = next(t for t in cases if t["id"] == "D7-T2")
    out2 = convolve(case2["input_matrix"], case2["kernel_T"])
    valid_region = case2["zero_pad_full_output_independent_recompute"]
    rows = len(valid_region)
    cols = len(valid_region[0])
    assert np.array_equal(out2["result"][:rows, :cols], np.array(valid_region))
