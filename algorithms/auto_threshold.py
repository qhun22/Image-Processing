import numpy as np

from core.validators import validate_matrix, validate_g, validate_p


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

    Ràng buộc G/P với dữ liệu ảnh:
    - G phải đủ lớn để chứa mọi mức xám xuất hiện trong ảnh (max(image) <= G-1), nếu không
      sẽ báo lỗi rõ ràng thay vì âm thầm bỏ sót pixel ngoài phạm vi [0, G-1].
    - P (nếu người dùng tự truyền, không dùng mặc định M*N) phải đúng bằng tổng số điểm ảnh
      có mức xám trong [0, G-1] (tức cumulative[G-1]) — không được khác, vì P đại diện cho
      đúng tập hợp pixel đang xét theo định nghăa "P là số điểm ảnh được xét" của tài liệu.

    Khi t(g) == P (lớp trên rỗng, tức mọi pixel đã nằm hết ở lớp dưới) hoặc t(g) == 0
    (lớp dưới rỗng), A(g)/B(g)/F(g) không xác định -> loại trừ g đó khỏi việc tìm argmax,
    KHÔNG gán Infinity hay 0 làm ngưỡng tối ưu một cách âm thầm.
    """
    img = validate_matrix(image)
    G = validate_g(G)
    M, N = img.shape
    total_pixels = M * N

    int_image = np.rint(img).astype(int)
    max_gray_in_image = int(int_image.max())
    if max_gray_in_image > G - 1:
        raise ValueError(
            f"G={G} không hợp lệ với ảnh đầu vào: ảnh có mức xám tối đa {max_gray_in_image}, "
            f"nhưng G chỉ cho phép xét mức xám tối đa {G - 1}. Mọi pixel có giá trị >= G sẽ bị "
            f"bỏ sót khỏi histogram nếu tiếp tục — hãy tăng G lên ít nhất {max_gray_in_image + 1}."
        )

    histogram = {g: int(np.sum(int_image == g)) for g in range(G)}

    cumulative = {}
    running = 0
    for g in range(G):
        running += histogram[g]
        cumulative[g] = running

    actual_total_in_range = cumulative[G - 1]

    if P is None:
        P = total_pixels
        if P != actual_total_in_range:
            # Không thể xảy ra vì đã validate max_gray_in_image <= G-1 ở trên, nhưng giữ lại
            # như một ràng buộc an toàn tường minh (defensive check).
            raise ValueError(
                "Lỗi nội bộ không mong đợi: tổng số điểm ảnh không khớp P mặc định."
            )
    else:
        P = validate_p(P, G)
        if P != actual_total_in_range:
            raise ValueError(
                f"P={P} không nhất quán với dữ liệu ảnh: tổng số điểm ảnh có mức xám trong "
                f"[0, {G - 1}] thực tế là {actual_total_in_range} (P phải bằng giá trị này theo "
                f"định nghĩa 'P là số điểm ảnh được xét' của tài liệu)."
            )

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
        # Loại trừ cả hai trường hợp suy biến: lớp dưới rỗng (t_g=0, nên A(g)=0 vô nghĩa
        # vì không có điểm ảnh nào để tính) và lớp trên rỗng (denom=0, chia cho 0).
        if t_g == 0 or denom == 0:
            A[g] = None
            F[g] = None
            skipped.append(g)
            continue
        A[g] = t_g / denom
        F[g] = A[g] * B[g] - 1

    candidates = {g: f for g, f in F.items() if f is not None}
    if not candidates:
        raise ValueError(
            "Không thể xác định ngưỡng: với mọi g trong [0, G-1], một trong hai lớp (dưới hoặc "
            "trên ngưỡng) đều rỗng, nên không thể tính phương sai liên lớp. Hãy kiểm tra lại G "
            "hoặc dữ liệu ảnh."
        )
    theta = max(candidates, key=candidates.get)

    # CHƯA XÁC NHẬN: tài liệu Dạng 2 (Bài giảng 3 / Slide Bài 3) chỉ trình bày cách tìm
    # ngưỡng theta tối ưu, KHÔNG có ví dụ số về quy tắc tạo ảnh nhị phân cụ thể (>= hay >)
    # cho riêng bước này. Quy tắc ">=" dưới đây được mượn từ Dạng 6 (Sobel), nơi ">=θ" đã
    # được xác nhận trực tiếp bằng ảnh PDF — KHÔNG được coi là đáp án duy nhất đã xác nhận
    # cho Dạng 2. Trả về trong "unconfirmed_binary_result" (không phải "result") để tránh
    # việc hiển thị như một kết quả chắc chắn.
    unconfirmed_binary_result = np.where(int_image >= theta, 1, 0)

    return {
        "threshold": theta,
        "F_theta": candidates[theta],
        "histogram": histogram,
        "cumulative": cumulative,
        "moment": moment,
        "A": A,
        "B": B,
        "F": F,
        "skipped_degenerate_classes": skipped,
        "unconfirmed_binary_result": unconfirmed_binary_result,
        "result_rule_confirmed": False,
        "steps": _build_steps(M, N, G, P, histogram, cumulative, moment, A, B, F, theta, skipped),
    }


def _build_steps(M, N, G, P, histogram, cumulative, moment, A, B, F, theta, skipped) -> list:
    steps = [
        f"Kích thước ảnh: {M}x{N}, G={G}, P={P}.",
    ]
    for g in range(G):
        if g in skipped:
            t_g = cumulative[g]
            reason = "t(g)=0 (lớp dưới rỗng)" if t_g == 0 else f"t(g)={t_g}=P (lớp trên rỗng)"
            steps.append(f"g={g}: {reason} -> A(g)/F(g) không xác định, loại khỏi tìm max.")
            continue
        steps.append(
            f"g={g}: h(g)={histogram[g]}, t(g)={cumulative[g]}, m(g)={moment[g]:.4f}, "
            f"A(g)={A[g]:.4f}, B(g)={B[g]:.4f}, F(g)={F[g]:.4f}"
        )
    steps.append(f"Ngưỡng tự động theta = {theta} (F(theta) lớn nhất = {F[theta]:.4f}).")
    steps.append(
        "LƯU Ý: quy tắc tạo ảnh nhị phân (I'(x,y)=1 nếu I(x,y)>=theta) CHƯA được tài liệu "
        "Dạng 2 xác nhận trực tiếp bằng ví dụ số — đây là giá trị tham khảo (mượn quy tắc từ "
        "Dạng 6 Sobel), không phải đáp án duy nhất đã được kiểm chứng."
    )
    return steps
