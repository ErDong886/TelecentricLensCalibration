"""单目标定的整体优化。"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def optimize_parameters(points_world, points_pixel, A_matrix, Ks_list, R_list, param_holder, verbose, log):
    """引入畸变模型并最小化重投影误差。"""
    num_images = len(points_world)

    # ========== 1. 准备初始参数 ==========
    # 内参初始值
    alpha = A_matrix[0, 0]
    beta = A_matrix[1, 1]
    gamma = A_matrix[0, 1]

    # 畸变系数初始值
    k1, k2 = 0.0, 0.0

    # 将旋转矩阵转为旋转向量（Rodrigues变换）
    rvecs = [cv2.Rodrigues(R)[0] for R in R_list]

    # 从 Ks_list 提取平移向量（最后一列的前两个值）
    tvecs = [Ks[:2, 2] for Ks in Ks_list]

    # 打包所有参数为一维向量
    params_vec = np.concatenate((
        [alpha, beta, gamma, k1, k2],
        np.array(rvecs).flatten(),
        np.array(tvecs).flatten()
    ))

    # ========== 2. 定义残差函数 ==========
    def residuals(params):
        """
        计算残差：observed - predicted
        符合公式(1)的远心相机投影模型
        """
        # 解包参数
        alpha, beta, gamma = params[0:3]
        k1, k2 = params[3:5]

        rvecs_flat = params[5 : 5 + num_images*3]
        tvecs_flat = params[5 + num_images*3 :]

        rvecs = rvecs_flat.reshape((num_images, 3))
        tvecs = tvecs_flat.reshape((num_images, 2))

        errors = []

        # 构建内参矩阵 ξ (符合图2)
        M_int = np.array([
            [alpha, gamma, 0, 0],
            [0,     beta,  0, 0],
            [0,     0,     0, 1]
        ])

        # 遍历每张图像
        for i in range(num_images):
            world_pts = points_world[i]  # (N, 3)
            pixel_pts_observed = points_pixel[i].reshape(-1, 2)  # (N, 2)

            # 构建外参矩阵 κ
            R, _ = cv2.Rodrigues(rvecs[i])
            tx, ty = tvecs[i]
            T = np.array([tx, ty, 0]).reshape(3, 1)  # tz = 0（远心相机特性）
            M_ext = np.vstack((np.hstack((R, T)), [0, 0, 0, 1]))  # 4x4

            # 将世界坐标转为齐次坐标 (4, N)
            num_pts = world_pts.shape[0]
            P_w = np.vstack((world_pts.T, np.ones(num_pts)))

            # 按照公式(1)投影：[u; v; 1] = ξ · κ · [Xw; Yw; Zw; 1]
            proj_homo = M_int @ M_ext @ P_w

            # 理想像素坐标（无畸变）
            u_ideal = proj_homo[0, :]
            v_ideal = proj_homo[1, :]

            # 应用径向畸变模型
            r_sq = u_ideal**2 + v_ideal**2
            distortion_factor = 1 + k1 * r_sq + k2 * r_sq**2
            u_pred = u_ideal * distortion_factor
            v_pred = v_ideal * distortion_factor

            pixel_pts_predicted = np.vstack((u_pred, v_pred)).T

            # 计算残差向量
            error = (pixel_pts_observed - pixel_pts_predicted).flatten()
            errors.extend(error)

        return np.array(errors)

    # ========== 3. 执行 Levenberg-Marquardt 优化 ==========
    log("   正在执行 Levenberg-Marquardt 优化...")
    # scipy的verbose参数：0=静默，1=显示，2=详细
    scipy_verbose = 1 if verbose else 0
    result = least_squares(
        residuals, 
        params_vec, 
        method='lm',  # Levenberg-Marquardt 算法
        verbose=scipy_verbose
    )

    # ========== 4. 解包优化后的参数 ==========
    optimized_params = result.x

    alpha_opt, beta_opt, gamma_opt = optimized_params[0:3]
    k1_opt, k2_opt = optimized_params[3:5]

    rvecs_opt = optimized_params[5:5+num_images*3].reshape((num_images, 3))
    tvecs_opt = optimized_params[5+num_images*3:].reshape((num_images, 2))

    # 将旋转向量转回旋转矩阵
    R_list_opt = [cv2.Rodrigues(rvec)[0] for rvec in rvecs_opt]

    # ========== 5. 更新参数容器 ==========
    param_holder.set_intrinsic(
        alpha=float(alpha_opt),
        beta=float(beta_opt),
        gamma=float(gamma_opt),
        k1=float(k1_opt),
        k2=float(k2_opt)
    )

    # 更新外参（使用第一组作为代表）
    param_holder.set_extrinsic(
        R=R_list_opt[0],
        T=np.array([tvecs_opt[0][0], tvecs_opt[0][1], 0])
    )

    log(f"   📐 优化后内参: alpha={alpha_opt:.4f}, beta={beta_opt:.4f}, gamma={gamma_opt:.4f}")
    log(f"   🔍 优化后畸变: k1={k1_opt:.6e}, k2={k2_opt:.6e}")

    return alpha_opt, beta_opt, gamma_opt, k1_opt, k2_opt, R_list_opt, tvecs_opt
