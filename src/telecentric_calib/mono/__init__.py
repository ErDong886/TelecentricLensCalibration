"""单相机标定流程。"""
from dataclasses import dataclass
import os

import numpy as np

from ..board import PatternBoard
from ..detect.fallback import detect_images, find_centers
from ..io import read_image
from .linear import solve_homography_and_intrinsics
from .optimize import optimize_parameters
from .pose import (
    construct_rotation_matrices,
    determine_r31_r32_signs,
    solve_known_intrinsic_pose,
    solve_r31_r32_magnitudes,
)


def make_logger(verbose):
    """详细输出开关。"""
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return log


def image_paths(image_dir):
    """按目录列举顺序收集图像，和原来的标定器一致。"""
    if os.path.isfile(image_dir):
        return [image_dir]
    paths = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    if not paths:
        raise ValueError(f"在目录 {image_dir} 中未找到图像文件")
    return paths


def board_from_holder(param_holder):
    """用参数容器里的标定板设置创建标定板。"""
    board_params = param_holder.get_board_dict()
    if board_params["dot_array_shape"] is None:
        raise ValueError("标定板参数未设置！请先设置 dot_array_shape")
    if board_params["dot_diameter"] is None:
        raise ValueError("标定板参数未设置！请先设置 dot_diameter")
    if board_params["dot_spacing"] is None:
        raise ValueError("标定板参数未设置！请先设置 dot_spacing")
    return PatternBoard(
        pattern_type=1,
        shape=board_params["dot_array_shape"],
        distance=board_params["dot_spacing"],
        pixel_size=(3.45, 3.45),
        radius=board_params["dot_diameter"] / 2,
    )


@dataclass
class MonoCalibration:
    """一台相机标定完成后的参数。"""
    param_holder: object
    image_size: tuple
    board: PatternBoard
    final_params: tuple
    verbose: bool

    @property
    def pattern_board(self):
        return self.board

    def known_intrinsic_calibration_single_image(self, image_path):
        """用已经确定的内参求一张图像的外参。"""
        if self.final_params is None:
            raise ValueError("final_params未确定，请先执行 calibrate_mono")
        image = read_image(image_path)
        pat_size = self.param_holder.get_board_dict()["dot_array_shape"]
        success, centers, _, _ = find_centers(image, pat_size)
        if not success:
            raise ValueError(f"角点检测失败: {image_path}")
        points_pixel = np.array(centers, dtype=np.float32).reshape(-1, 2)
        points_world = self.board.generate_world_points_grid()
        alpha, beta, gamma = self.final_params[:3]
        return solve_known_intrinsic_pose(
            points_world,
            points_pixel,
            alpha,
            beta,
            gamma,
            self.image_size,
            make_logger(self.verbose),
            image_path,
        )


def calibrate(image_dir, param_holder, verbose=True):
    """检测圆点并完成单相机标定。"""
    paths = image_paths(image_dir)
    first = read_image(paths[0])
    image_size = (first.shape[1], first.shape[0])
    board = board_from_holder(param_holder)
    log = make_logger(verbose)
    points_world, points_pixel = detect_images(paths, board, log)
    _, intrinsic, truncated = solve_homography_and_intrinsics(points_world, points_pixel)
    magnitudes = solve_r31_r32_magnitudes(truncated)
    signs = determine_r31_r32_signs(points_world, points_pixel, image_size, log)
    rotations = construct_rotation_matrices(truncated, magnitudes, signs)
    final_params = optimize_parameters(
        points_world,
        points_pixel,
        intrinsic,
        truncated,
        rotations,
        param_holder,
        verbose,
        log,
    )
    return MonoCalibration(param_holder, image_size, board, final_params, verbose)
