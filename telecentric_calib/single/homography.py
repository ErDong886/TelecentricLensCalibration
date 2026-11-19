"""由圆点对应关系求单应矩阵、远心内参和截断外参。"""
import cv2
import numpy as np


def calculate_homography_matrix(points_pixel, points_world):
    """
    使用 cv2.findHomography 的 DLT（method=0）计算每张图的单应矩阵。

    points_pixel: (n_images, n_points, 2)
    points_world: (n_images, n_points, 3)，只使用 X、Y
    """
    homographies = []
    for index, (pixel_points, world_points) in enumerate(zip(points_pixel, points_world)):
        src_pts = np.array(world_points)[:, :2].astype(np.float32)
        dst_pts = np.array(pixel_points).astype(np.float32)
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if homography is None:
            raise ValueError(f"图像 {index} 的单应性矩阵计算失败！")
        if homography[2, 2] != 0:
            homography = homography / homography[2, 2]
        homographies.append(homography)
    return homographies


def calculate_intrinsic_matrix(homographies):
    """
    由一组单应矩阵线性求解远心内参 alpha、beta、gamma。

    实现对应 Chen 等 (2014) 与 Li 等 (2024) 的线性标定。至少需要 4 张图。
    """
    num_h = len(homographies)
    if num_h < 4:
        raise ValueError(f"至少需要4个单应性矩阵进行求解，但只提供了 {num_h} 个。")

    design = np.zeros((num_h, 4))
    observation = np.zeros((num_h, 1))
    for index, homography in enumerate(homographies):
        h11, h12, _ = homography[0]
        h21, h22, _ = homography[1]
        design[index] = [
            1.0,
            -(h21 ** 2 + h22 ** 2),
            -(h11 ** 2 + h12 ** 2),
            2 * (h11 * h21 + h12 * h22),
        ]
        observation[index] = -(h11 * h22 - h12 * h21) ** 2

    try:
        solution, _, _, _ = np.linalg.lstsq(design, observation, rcond=None)
        _l1, l2, l3, l4 = solution.flatten()
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"求解线性方程组失败，请检查H矩阵是否有效。错误: {exc}") from exc

    alpha_numerator = l2 * l3 - l4 ** 2
    if alpha_numerator <= 0:
        raise ValueError(f"alpha分子为非正数({alpha_numerator:.4f})，无法计算alpha。")
    if l3 <= 0:
        raise ValueError(f"l3值为非正数({l3:.4f})，无法计算beta。这通常意味着输入数据有误或存在退化配置。")

    alpha = np.sqrt(alpha_numerator / l3)
    beta = np.sqrt(l3)
    gamma = l4 / beta
    intrinsic = np.array([
        [alpha, gamma, 0],
        [0, beta, 0],
        [0, 0, 1],
    ])
    return intrinsic


def calculate_truncated_extrinsic_matrix(intrinsic, homographies):
    """按 Ks = A^{-1} H 计算每张图的截断外参。"""
    try:
        intrinsic_inverse = np.linalg.inv(intrinsic)
    except np.linalg.LinAlgError as exc:
        raise ValueError("内参矩阵A是奇异矩阵，无法求逆。") from exc
    return [intrinsic_inverse @ homography for homography in homographies]


def solve_h_a_ks(points_world, points_pixel):
    """Step 1：求解 H、内参 A 和截断外参 Ks。"""
    homographies = calculate_homography_matrix(points_pixel, points_world)
    intrinsic = calculate_intrinsic_matrix(homographies)
    truncated = calculate_truncated_extrinsic_matrix(intrinsic, homographies)
    return homographies, intrinsic, truncated
