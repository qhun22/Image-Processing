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


def test_no_false_positive_tie_when_tied_values_are_identical():
    # Cửa sổ có các giá trị đồng khoảng cách nhưng BẰNG NHAU (toàn số 0 do padding) không
    # phải là một tie thực sự (chọn phần tử nào trong số các giá trị giống hệt nhau cũng
    # cho cùng kết quả) -> không được báo cờ tie_break_used một cách thừa.
    image = [[0, 2], [2, 0]]
    out = k_nearest_mean_filter(image, window_size=3, threshold=100, k=2)
    assert out["tie_break_used_anywhere"] is False


def test_genuine_tie_between_two_different_values_at_k_boundary():
    # Thiết kế ma trận sao cho tại ranh giới chọn k, có đúng 2 giá trị KHÁC NHAU (3 và 7)
    # cùng khoảng cách (=2) tới pixel trung tâm (giá trị 5). Với k=2: chọn center (dist 0)
    # + một trong {3,7} (dist 2, tie thật). Hai lựa chọn cho TBk khác hẳn nhau:
    #   chọn 3 -> TBk=(5+3)/2=4
    #   chọn 7 -> TBk=(5+7)/2=6
    # Đây là bằng chứng trực tiếp rằng việc chọn sai gây ra hậu quả khác biệt lên kết quả,
    # không phải một chi tiết vô hại.
    image = [
        [9, 9, 9],
        [9, 5, 9],
        [3, 9, 7],
    ]
    out = k_nearest_mean_filter(image, window_size=3, threshold=0, k=2)
    log = next(l for l in out["window_logs"] if l["position"] == (1, 1))

    assert log["tie_break_used"] is True
    assert out["tie_break_used_anywhere"] is True

    # Quy tắc tạm thời hiện tại (thứ tự quét trái->phải, trên->xuống) chọn giá trị 3
    # (xuất hiện ở vị trí (2,0), trước 7 ở vị trí (2,2)) -> TBk=4. Đây LÀ MỘT GIẢ ĐỊNH
    # CHƯA CÓ CĂN CỨ TỪ TÀI LIỆU, test này chỉ xác nhận hành vi hiện tại của code (để
    # phát hiện thay đổi không cố ý), không khẳng định đây là đáp án đúng duy nhất.
    assert log["k_nearest_values"] == [5.0, 3.0]
    assert log["tbk_rounded"] == 4
    # Nếu tie được giải quyết theo hướng ngược lại (chọn 7), kết quả sẽ là 6 - khác hẳn.
    alternative_tbk = round((5 + 7) / 2)
    assert alternative_tbk != log["tbk_rounded"], "Tie thực sự phải tạo ra 2 kết quả khác nhau rõ ràng."


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
