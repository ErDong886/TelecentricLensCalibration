"""带径向畸变的远心投影，以及整体 Levenberg-Marquardt 优化。"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def _residuals(params, points_world, points_pixel):
    """观测像素减去公式 (1) 的预测像素。"""
    num_images = len(points_world)
    alpha, beta, gamma = params[0:3]
    k1, k2 = params[3:5]
    rvecs = params[5:5 + num_images * 3].reshape((num_images, 3))
    tvecs = params[5 + num_images * 3:].reshape((num_images, 2))
    intrinsic = np.array([
        [alpha, gamma, 0, 0],
        [0, beta, 0, 0],
        [0, 0, 0, 1],
    ])

    errors = []
    for index in range(num_images):
        world_points = points_world[index]
        observed = points_pixel[index].reshape(-1, 2)
        rotation, _ = cv2.Rodrigues(rvecs[index])
        tx, ty = tvecs[index]
        translation = np.array([tx, ty, 0]).reshape(3, 1)
        extrinsic = np.vstack((np.hstack((rotation, translation)), [0, 0, 0, 1]))
        num_points = world_points.shape[0]
        homogeneous = np.vstack((world_points.T, np.ones(num_points)))
        projected = intrinsic @ extrinsic @ homogeneous
        u_ideal = projected[0, :]
        v_ideal = projected[1, :]
        radius_sq = u_ideal ** 2 + v_ideal ** 2
        distortion_factor = 1 + k1 * radius_sq + k2 * radius_sq ** 2
        predicted = np.vstack((u_ideal * distortion_factor, v_ideal * distortion_factor)).T
        errors.extend((observed - predicted).flatten())
    return np.array(errors)


def global_optimize(points_world, points_pixel, _homographies, intrinsic, truncated_extrinsics, rotations, verbose, log):
    """Step 5：以重投影误差优化内参、畸变和每张图的外参。"""
    num_images = len(points_world)
    alpha = intrinsic[0, 0]
    beta = intrinsic[1, 1]
    gamma = intrinsic[0, 1]
    rvecs = [cv2.Rodrigues(rotation)[0] for rotation in rotations]
    tvecs = [truncated[:2, 2] for truncated in truncated_extrinsics]
    initial = np.concatenate((
        [alpha, beta, gamma, 0.0, 0.0],
        np.array(rvecs).flatten(),
        np.array(tvecs).flatten(),
    ))

    log("   正在执行 Levenberg-Marquardt 优化...")
    result = least_squares(
        _residuals,
        initial,
        args=(points_world, points_pixel),
        method="lm",
        verbose=1 if verbose else 0,
    )
    optimized = result.x
    alpha_opt, beta_opt, gamma_opt = optimized[0:3]
    k1_opt, k2_opt = optimized[3:5]
    rvecs_opt = optimized[5:5 + num_images * 3].reshape((num_images, 3))
    tvecs_opt = optimized[5 + num_images * 3:].reshape((num_images, 2))
    rotations_opt = [cv2.Rodrigues(rvec)[0] for rvec in rvecs_opt]
    return alpha_opt, beta_opt, gamma_opt, k1_opt, k2_opt, rotations_opt, tvecs_opt
