"""单目标定的线性部分：单应矩阵、内参和截断外参。"""
import cv2
import numpy as np


def calculate_homography_matrix(points_pixel, points_world):
    """
    使用cv2.findHomography，method0代表DLT算法，计算单应性矩阵H

    参数:
    points_pixel: 所有照片的像素坐标点 shape: (n_images, n_points_per_image, 2)
    points_world: 所有照片的世界坐标点 shape: (n_images, n_points_per_image, 2) (假设Z=0)

    返回:
    H_matrices: 每张图片的单应性矩阵列表
    """

    H_matrices = []

    for i in range(len(points_pixel)):
        pixel_points = points_pixel[i]  # 当前图像的像素点
        world_points = points_world[i]  # 当前图像的世界坐标点

        # 只取世界坐标的前两个分量（X, Y），因为Z=0
        src_pts = np.array(world_points)[:, :2].astype(np.float32)
        dst_pts = np.array(pixel_points).astype(np.float32)

        # cv2.findHomography 要求输入为 (N,2) 的float32数组
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)  # 直接线性变换

        if H is None:
            raise ValueError(f"图像 {i} 的单应性矩阵计算失败！")

        # 归一化（通常让H[2,2] = 1）
        if H[2, 2] != 0:
            H = H / H[2, 2]

        H_matrices.append(H)

    return H_matrices

def calculate_intrinsic_matrix(H_matrices):
    """
    根据一系列单应性矩阵，计算并返回远心相机的内参 alpha, beta, gamma。
    该实现基于Chen等(2014)和Li等(2024)论文中描述的线性标定方法。

    :param H_matrices: 一个列表，其中每个元素都是一个3x3的单应性矩阵(Numpy array)。
    :return: 一个元组 (alpha, beta, gamma)，包含了计算出的三个内参。
    :raises ValueError: 如果输入的单应性矩阵数量少于4个，或者在计算中出现数学错误（如对负数开方）。
    """

    num_H = len(H_matrices)
    if num_H < 4:
        raise ValueError(f"至少需要4个单应性矩阵进行求解，但只提供了 {num_H} 个。")

    # 初始化G矩阵和w向量
    G = np.zeros((num_H, 4))
    w = np.zeros((num_H, 1))

    # 步骤1: 构建线性方程组 GL = w
    for i, H in enumerate(H_matrices):
        # 提取H矩阵的元素用于计算
        h11, h12, _ = H[0]
        h21, h22, _ = H[1]

        # 根据公式(14)构建G矩阵的一行
        g1 = 1.0
        g2 = -(h21**2 + h22**2)
        g3 = -(h11**2 + h12**2)
        g4 = 2 * (h11 * h21 + h12 * h22)

        G[i] = [g1, g2, g3, g4]

        # 构建w向量的一个元素
        w[i] = -(h11 * h22 - h12 * h21)**2

    # 步骤2: 使用最小二乘法求解线性系统
    try:
        L, _, _, _ = np.linalg.lstsq(G, w, rcond=None)
        l1, l2, l3, l4 = L.flatten()
    except np.linalg.LinAlgError as e:
        raise ValueError(f"求解线性方程组失败，请检查H矩阵是否有效。错误: {e}")

    # 步骤3: 从L中分解出alpha, beta, gamma
    # 计算 α = sqrt((l2*l3 - l4²)/l3)
    alpha_numerator = l2 * l3 - l4**2
    if alpha_numerator <= 0:
        raise ValueError(f"alpha分子为非正数({alpha_numerator:.4f})，无法计算alpha。")

    alpha = np.sqrt(alpha_numerator / l3)

    # 计算 β = sqrt(l3)
    if l3 <= 0:
        raise ValueError(f"l3值为非正数({l3:.4f})，无法计算beta。这通常意味着输入数据有误或存在退化配置。")

    beta = np.sqrt(l3)
    gamma = l4/beta
    A_matrix = np.array([[alpha, gamma, 0],
                         [0, beta, 0],
                         [0, 0, 1]])
    return A_matrix

def calculate_truncated_extrinsic_matrix(A_matrix, H_matrices):
    """
    根据相机的内参矩阵A和一系列单应性矩阵，
    计算并返回每个姿态对应的截断外参矩阵K_s。

    此函数依据公式 Ks = A⁻¹ * H 实现。

    :param A_matrix: 相机内参矩阵 (3x3 Numpy array)
    :param H_matrices: 一个列表，其中每个元素都是一个3x3的单应性矩阵 (Numpy array)。
    :return: 一个列表，其中每个元素都是一个3x3的截断外参矩阵K_s。
    :raises ValueError: 如果内参矩阵A是奇异的（不可逆）。
    """

    # 计算内参矩阵A的逆矩阵 A⁻¹
    try:
        A_inv = np.linalg.inv(A_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("内参矩阵A是奇异矩阵，无法求逆。")

    Ks_list = []

    # 遍历每一个单应性矩阵H，并计算对应的K_s
    for H in H_matrices:
        # 根据公式计算 K_s = A⁻¹ * H
        Ks = A_inv @ H
        Ks_list.append(Ks)
    return Ks_list


def solve_homography_and_intrinsics(points_world, points_pixel):
    """求解每张图的单应矩阵、内参 A 和截断外参 Ks。"""
    h_matrices = calculate_homography_matrix(points_pixel, points_world)
    a_matrix = calculate_intrinsic_matrix(h_matrices)
    ks_list = calculate_truncated_extrinsic_matrix(a_matrix, h_matrices)
    return h_matrices, a_matrix, ks_list
