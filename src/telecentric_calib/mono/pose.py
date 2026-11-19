"""单目外参：r31/r32 的大小与符号、旋转矩阵，以及已知内参时的单张姿态。"""
import cv2
import numpy as np
from scipy.optimize import fsolve


def solve_r31_r32_magnitudes(ks_list):
    """Step 2: 依据正交性和模长相等约束，求解r31和r32的绝对值"""
    magnitudes_list = []

    for Ks in ks_list:
        # 1. 从Ks矩阵中提取已知的外参元素
        r11, r21, _ = Ks[:, 0]
        r12, r22, _ = Ks[:, 1]

        # 2. 定义基于新约束的方程组
        # 正交性约束: r11*r12 + r21*r22 + r31*r32 = 0
        # 模长相等约束: r11^2+r21^2+r31^2 = r12^2+r22^2+r32^2
        def equations(p):
            eq1 = r11*r12 + r21*r22 + p[0]*p[1]
            eq2 = (r11**2 + r21**2 + p[0]**2) - (r12**2 + r22**2 + p[1]**2)
            return (eq1, eq2)

        # 3. 求解方程组
        r31_solution, r32_solution = fsolve(equations, (0.01,0.01))

        # 4. 提取解的绝对值
        abs_r31 = np.abs(r31_solution)
        abs_r32 = np.abs(r32_solution)

        magnitudes_list.append((abs_r31, abs_r32))

    return magnitudes_list


def determine_r31_r32_signs(points_world, points_pixel, image_size, log):
    """Step 3: 按照Li Hai等人(2024)论文中所描述的，使用针孔模型确定r31和r32的符号"""

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_world,
        points_pixel,
        image_size,
        None,
        None
    )

    if not ret:
        raise RuntimeError("针孔模型标定失败，无法获取参考符号")
    else:
        # 详细调试信息仅在verbose模式下输出
        log(f"重投影误差为：{ret}，内参矩阵为：{mtx}，畸变系数为：{dist}，旋转向量rvecs为：{rvecs}，平移向量tvecs为：{tvecs}")

    reference_signs_list = []
    for rvec in rvecs:
        # 将旋转向量转换为旋转矩阵，获取针孔模型的旋转矩阵作为符号参考
        R_pinhole, _ = cv2.Rodrigues(rvec)

        sign_r31 = np.sign(R_pinhole[2, 0])
        sign_r32 = np.sign(R_pinhole[2, 1])

        if sign_r31 == 0:
            sign_r31 = 1.0
        if sign_r32 == 0:
            sign_r32 = 1.0

        reference_signs_list.append((sign_r31, sign_r32))

    return reference_signs_list


def construct_rotation_matrices(ks_list, magnitudes_list, signs_list):
    """Step 4: 通过向量归一化和交叉积求出完整的旋转矩阵R，并利用SVD估计出最优的旋转矩阵Ropt"""
    R_list = []
    for Ks, (abs_r31, abs_r32), (sign_r31, sign_r32) in zip(ks_list, magnitudes_list, signs_list):
        # 计算正确的r31和r32
        correct_r31 = abs_r31 * sign_r31
        correct_r32 = abs_r32 * sign_r32

        r11, r21, _ = Ks[:, 0]
        r12, r22, _ = Ks[:, 1]

        rx = np.array([r11, r21, correct_r31])
        ry = np.array([r12, r22, correct_r32])

        # 通过叉乘得到rz
        rz = np.cross(rx, ry)

        # 构造旋转矩阵R，其列为rx, ry, rz
        R_approx = np.column_stack([rx, ry, rz])

        # 使用SVD方法对R进行强制正交化
        U, _, Vt = np.linalg.svd(R_approx)

        # 强制正交化：R_ortho = U @ Vt
        R_ortho = U @ Vt

        R_list.append(R_ortho)

    return R_list


def solve_known_intrinsic_pose(points_world, points_pixel, alpha, beta, gamma, image_size, log, image_path):
    """用已知内参求单张图像的旋转和平移。"""
    src_pts = np.array(points_world)[:, :2].astype(np.float32)
    dst_pts = points_pixel.reshape(-1, 2).astype(np.float32)

    homography, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    if homography is None:
        raise ValueError(f"单应性矩阵H计算失败: {image_path}")
    if homography[2, 2] != 0:
        homography = homography / homography[2, 2]

    intrinsic = np.array([
        [alpha, gamma, 0],
        [0, beta, 0],
        [0, 0, 1],
    ])
    truncated = np.linalg.inv(intrinsic) @ homography
    r11, r21, _ = truncated[:, 0]
    r12, r22, _ = truncated[:, 1]

    def equations(unknown):
        orthogonal = r11 * r12 + r21 * r22 + unknown[0] * unknown[1]
        equal_norm = (r11**2 + r21**2 + unknown[0]**2) - (r12**2 + r22**2 + unknown[1]**2)
        return (orthogonal, equal_norm)

    r31_solution, r32_solution = fsolve(equations, (0.01, 0.01))
    abs_r31 = np.abs(r31_solution)
    abs_r32 = np.abs(r32_solution)

    ret, _, _, rvecs, _ = cv2.calibrateCamera(
        [points_world],
        [points_pixel],
        image_size,
        None,
        None,
    )
    if not ret:
        log("   ⚠️  针孔模型标定失败，使用默认符号")
        sign_r31, sign_r32 = 1.0, 1.0
    else:
        pinhole_rotation, _ = cv2.Rodrigues(rvecs[0])
        sign_r31 = np.sign(pinhole_rotation[2, 0]) if pinhole_rotation[2, 0] != 0 else 1.0
        sign_r32 = np.sign(pinhole_rotation[2, 1]) if pinhole_rotation[2, 1] != 0 else 1.0

    correct_r31 = abs_r31 * sign_r31
    correct_r32 = abs_r32 * sign_r32
    rx = np.array([r11, r21, correct_r31])
    ry = np.array([r12, r22, correct_r32])
    rz = np.cross(rx, ry)
    approximate = np.column_stack([rx, ry, rz])
    left_singular, _, right_singular = np.linalg.svd(approximate)
    rotation = left_singular @ right_singular
    translation = np.array([truncated[0, 2], truncated[1, 2]])
    rvec, _ = cv2.Rodrigues(rotation)

    log("   ✅ 外参求解成功")
    log(f"   旋转矩阵R:\n{rotation}")
    log(f"   旋转向量rvec: {rvec.T}")
    log(f"   平移向量tvec: {translation}")
    log(f"   det(R): {np.linalg.det(rotation):.6f}")
    return rvec, translation, rotation
