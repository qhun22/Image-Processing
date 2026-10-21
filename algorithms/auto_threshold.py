import math

import numpy as np

from core.validators import validate_matrix, validate_g


def auto_threshold(image, G, P=None) -> dict:
    """
    Tách ngưỡng tự động theo công thức mômen quán tính (Bài giảng 3.pdf trang 3-4;
    Slide Bài 3 trang 9-13):
        t(g) = số điểm ảnh có mức xám <= g
        m(g) = sum(i*h(i), i=0..g) / t(g)
        A(g) = t(g) / (P - t(g))
        B(g) = [m(g) - m(G-1)]^2
        F(g) = A(g)*B(g) - 1
        theta = argmax F(g), với g <= G-1

    Đã chứng minh bằng đại số (docs/audit/corrections_v5.json): A(g)*B(g) tương đương
    phương sai liên lớp Otsu (omega0*omega1*(mu0-mu1)^2), nhưng triển khai dùng ĐÚNG công
    thức tài liệu viết ra (không thay bằng công thức Otsu "chuẩn" hay cv2.THRESH_OTSU) để
    các bước trung gian A(g), B(g) khớp với lời giải giáo trình.

    Khi t(g) == P (lớp trên rỗng), A(g) chia cho 0 -> loại trừ g đó khỏi việc tìm argmax
    (không gán Infinity làm ngưỡng tối ưu một cách âm thầm).
    """
    img = validate_matrix(image)
    G = validate_g(G)
    M, N = img.shape
    total_pixels = M * N
    P = total_pixels if P is None else int(P)
    if P <= 0:
        raise ValueError("P phải là số nguyên dương.")

    int_image = np.rint(img).astype(int)

    histogram = {g: int(np.sum(int_image == g)) for g in range(G)}

    cumulative = {}
    running = 0
    for g in range(G):
        running += histogram[g]
        cumulative[g] = running

    moment = {}
    running_moment = 0
    for g in range(G):
        running_moment += g * histogram[g]
        moment[g] = running_moment / cumulative[g] if cumulative[g] > 0 else 0.0

    mu_global = moment[G - 1]

    A = {}
    B = {}
    F = {}
    skipped = []
    for g in range(G):
        t_g = cumulative[g]
        denom = P - t_g
        B[g] = (moment[g] - mu_global) ** 2
        if denom == 0:
            A[g] = None
            F[g] = None
            skipped.append(g)
            continue
        A[g] = t_g / denom
        F[g] = A[g] * B[g] - 1

    candidates = {g: f for g, f in F.items() if f is not None}
    if not candidates:
        raise ValueError(
            "Không thể xác định ngưỡng: mọi giá trị g đều khiến lớp trên rỗng (P - t(g) = 0)."
        )
    theta = max(candidates, key=candidates.get)

    # CHƯA XÁC NHẬN: tài liệu Dạng 2 (Bài giảng 3 / Slide Bài 3) chỉ trình bày cách tìm
    # ngưỡng theta tối ưu, KHÔNG có ví dụ số về quy tắc tạo ảnh nhị phân cụ thể (>= hay >)
    # cho riêng bước này. Quy tắc ">=" dưới đây được mượn từ Dạng 6 (Sobel), nơi ">=θ" đã
    # được xác nhận trực tiếp bằng ảnh PDF — áp dụng tạm thời cho nhất quán, nhưng PHẢI coi
    # là giả định chưa có bằng chứng trực tiếp cho Dạng 2, cần người dùng xác nhận.
    binary_result = np.where(int_image >= theta, 1, 0)

    return {
        "threshold": theta,
        "F_theta": candidates[theta],
        "histogram": histogram,
        "cumulative": cumulative,
        "moment": moment,
        "A": A,
        "B": B,
        "F": F,
        "skipped_due_to_division_by_zero": skipped,
        "result": binary_result,
        "result_rule_confirmed": False,
        "steps": _build_steps(M, N, G, P, histogram, cumulative, moment, A, B, F, theta, skipped),
    }


def _build_steps(M, N, G, P, histogram, cumulative, moment, A, B, F, theta, skipped) -> list:
    steps = [
        f"Kích thước ảnh: {M}x{N}, G={G}, P={P}.",
    ]
    for g in range(G):
        if g in skipped:
            steps.append(f"g={g}: t(g)={cumulative[g]} = P -> A(g) không xác định (chia cho 0), loại khỏi tìm max.")
            continue
        steps.append(
            f"g={g}: h(g)={histogram[g]}, t(g)={cumulative[g]}, m(g)={moment[g]:.4f}, "
            f"A(g)={A[g]:.4f}, B(g)={B[g]:.4f}, F(g)={F[g]:.4f}"
        )
    steps.append(f"Ngưỡng tự động theta = {theta} (F(theta) lớn nhất = {F[theta]:.4f}).")
    steps.append(
        "Ảnh nhị phân (CHƯA XÁC NHẬN từ tài liệu Dạng 2, mượn quy tắc từ Dạng 6 Sobel): "
        "I'(x,y) = 1 nếu I(x,y) >= theta, ngược lại = 0."
    )
    return steps
