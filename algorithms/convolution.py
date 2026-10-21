from core.validators import validate_matrix, validate_kernel
from core.matrix_utils import convolve_formula_2_1


def convolve(image, kernel) -> dict:
    """
    Nhân chập theo công thức 2.1 (BÀI 4.pdf trang 1-4):
        I⊗T(x,y) = sum_{i=0}^{m-1} sum_{j=0}^{n-1} I(x+i, y+j) * T(i,j)

    Không lật kernel. Zero-padding. Output cùng kích thước ảnh gốc.
    Đã verify 100% bằng 2 ví dụ số có đáp án (mặt nạ 2x2).

    ANCHOR CHO MẶT NẠ LẺ (3x3+) CHƯA ĐƯỢC XÁC NHẬN ĐẦY ĐỦ TỪ TÀI LIỆU (xem
    docs/audit/corrections_v5.json): công thức 2.1 vẫn được áp dụng NGUYÊN VĂN làm quy tắc
    chính thức duy nhất có bằng chứng cho Dạng 7 (không mượn quy tắc anchor tâm của toán tử
    Laplace/Sobel — đó là các chương riêng biệt, không phải Dạng 7). Kết quả trả về có cờ
    anchor_rule_confirmed=False khi kernel có kích thước lẻ >= 3 để người dùng biết đây là
    trường hợp cần xác nhận thêm, KHÔNG phải một đáp án chắc chắn duy nhất.
    """
    img = validate_matrix(image)
    ker = validate_kernel(kernel)

    m, n = ker.shape
    anchor_confirmed = not (m % 2 == 1 and m >= 3) and not (n % 2 == 1 and n >= 3)

    result = convolve_formula_2_1(img, ker)

    return {
        "result": result,
        "anchor_rule_confirmed": anchor_confirmed,
        "steps": _build_steps(img, ker, result, anchor_confirmed),
    }


def _build_steps(img, ker, result, anchor_confirmed) -> list:
    M, N = img.shape
    m, n = ker.shape
    steps = [
        f"Công thức 2.1: I⊗T(x,y) = sum(i=0..{m-1}, j=0..{n-1}) I(x+i,y+j)*T(i,j), zero-padding.",
    ]
    if not anchor_confirmed:
        steps.append(
            f"CẢNH BÁO: mẫu T kích thước {m}x{n} (lẻ, >=3) — quy tắc anchor cho kích thước này "
            "CHƯA được xác nhận đầy đủ bằng ví dụ số độc lập trong tài liệu. Kết quả dưới đây "
            "áp dụng công thức 2.1 nguyên văn (không dịch chuyển offset), là lựa chọn có "
            "bằng chứng tốt nhất hiện có, nhưng CẦN xác nhận thêm trước khi coi là đáp án cuối."
        )
    for x in range(M):
        for y in range(N):
            terms = []
            for i in range(m):
                for j in range(n):
                    xi, yj = x + i, y + j
                    i_val = img[xi, yj] if (0 <= xi < M and 0 <= yj < N) else 0
                    terms.append(f"{i_val}*{ker[i,j]}")
            steps.append(f"I'({x},{y}) = {' + '.join(terms)} = {result[x,y]:.4g}")
    return steps
