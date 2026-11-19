"""由截断外参恢复 r31、r32 和旋转矩阵。"""
import cv2
import numpy as np
from scipy.optimize import fsolve


def _r31_r32_equations(known_columns, unknowns):
    r11, r21, r12, r22 = known_columns
    r31, r32 = unknowns
    orthogonality = r11 * r12 + r21 * r22 + r31 * r32
    equal_norm = (r11 ** 2 + r21 ** 2 + r31 ** 2) - (r12 ** 2 + r22 ** 2 + r32 ** 2)
    return (orthogonality, equal_norm)


def solve_r31_r32_magnitudes(truncated_extrinsics):
    """Step 2：用正交性和等模长约束求 r31、r32 的绝对值。"""
    magnitudes = []
    for truncated in truncated_extrinsics:
        r11, r21, _ = truncated[:, 0]
        r12, r22, _ = truncated[:, 1]
        known = (r11, r21, r12, r22)
        r31_solution, r32_solution = fsolve(
            lambda unknowns, known=known: _r31_r32_equations(known, unknowns),
            (0.01, 0.01),
        )
        magnitudes.append((np.abs(r31_solution), np.abs(r32_solution)))
    return magnitudes


def determine_r31_r32_signs(points_world, points_pixel, image_size, log):
    """Step 3：用针孔模型的旋转矩阵给 r31、r32 定符号。"""
    reprojection_error, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera(
        points_world,
        points_pixel,
        image_size,
        None,
        None,
    )
    if not reprojection_error:
        raise RuntimeError("针孔模型标定失败，无法获取参考符号")
    log(
        f"重投影误差为：{reprojection_error}，内参矩阵为：{intrinsic}，"
        f"畸变系数为：{distortion}，旋转向量rvecs为：{rvecs}，平移向量tvecs为：{tvecs}"
    )

    signs = []
    for rvec in rvecs:
        pinhole_rotation, _ = cv2.Rodrigues(rvec)
        sign_r31 = np.sign(pinhole_rotation[2, 0])
        sign_r32 = np.sign(pinhole_rotation[2, 1])
        if sign_r31 == 0:
            sign_r31 = 1.0
        if sign_r32 == 0:
            sign_r32 = 1.0
        signs.append((sign_r31, sign_r32))
    return signs


def construct_rotation_matrices(truncated_extrinsics, magnitudes, signs):
    """Step 4：补上第三行，再用 SVD 把旋转矩阵正交化。"""
    rotations = []
    for truncated, (abs_r31, abs_r32), (sign_r31, sign_r32) in zip(
        truncated_extrinsics, magnitudes, signs
    ):
        r11, r21, _ = truncated[:, 0]
        r12, r22, _ = truncated[:, 1]
        column_x = np.array([r11, r21, abs_r31 * sign_r31])
        column_y = np.array([r12, r22, abs_r32 * sign_r32])
        column_z = np.cross(column_x, column_y)
        approximate = np.column_stack([column_x, column_y, column_z])
        left, _, right = np.linalg.svd(approximate)
        rotations.append(left @ right)
    return rotations


def solve_pose_with_known_intrinsics(points_world, points_pixel, alpha, beta, gamma, image_size, image_path, log):
    """已知内参时，求一张图的旋转和平移。"""
    src_pts = np.array(points_world)[:, :2].astype(np.float32)
    dst_pts = np.array(points_pixel).reshape(-1, 2).astype(np.float32)
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
    known = (r11, r21, r12, r22)
    r31_solution, r32_solution = fsolve(
        lambda unknowns, known=known: _r31_r32_equations(known, unknowns),
        (0.01, 0.01),
    )
    abs_r31 = np.abs(r31_solution)
    abs_r32 = np.abs(r32_solution)

    success, _mtx, _dist, rvecs, _tvecs = cv2.calibrateCamera(
        [points_world],
        [points_pixel],
        image_size,
        None,
        None,
    )
    if not success:
        log("   ⚠️  针孔模型标定失败，使用默认符号")
        sign_r31, sign_r32 = 1.0, 1.0
    else:
        pinhole_rotation, _ = cv2.Rodrigues(rvecs[0])
        sign_r31 = np.sign(pinhole_rotation[2, 0]) if pinhole_rotation[2, 0] != 0 else 1.0
        sign_r32 = np.sign(pinhole_rotation[2, 1]) if pinhole_rotation[2, 1] != 0 else 1.0

    column_x = np.array([r11, r21, abs_r31 * sign_r31])
    column_y = np.array([r12, r22, abs_r32 * sign_r32])
    column_z = np.cross(column_x, column_y)
    approximate = np.column_stack([column_x, column_y, column_z])
    left, _, right = np.linalg.svd(approximate)
    rotation = left @ right
    translation = np.array([truncated[0, 2], truncated[1, 2]])
    rvec, _ = cv2.Rodrigues(rotation)

    log("   ✅ 外参求解成功")
    log(f"   旋转矩阵R:\n{rotation}")
    log(f"   旋转向量rvec: {rvec.T}")
    log(f"   平移向量tvec: {translation}")
    log(f"   det(R): {np.linalg.det(rotation):.6f}")
    return rvec, translation, rotation
