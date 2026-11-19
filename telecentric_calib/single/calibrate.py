"""单目远心标定。这里只按六个步骤调用，计算在同目录的其他模块里。"""
import os

import numpy as np

from ..detect import detect_calibration_image
from ..io import (
    build_camera_params_dict,
    build_ck_matrix_dict,
    list_calibration_images,
    read_image,
    save_json,
)
from ..params import CalibParamHolder, pattern_board_from_holder
from .extrinsics import (
    construct_rotation_matrices,
    determine_r31_r32_signs,
    solve_pose_with_known_intrinsics,
    solve_r31_r32_magnitudes,
)
from .homography import solve_h_a_ks
from .optimize import global_optimize


class TelecentricCalibrator:
    """对一个图像目录做单目远心标定。"""

    def __init__(self, param_holder: CalibParamHolder, image_dir: str, verbose: bool = True):
        self.param_holder = param_holder
        self.image_dir = image_dir
        self.verbose = verbose
        self.final_params = None
        image_files = [
            name for name in os.listdir(image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        if not image_files:
            raise ValueError(f"在目录 {image_dir} 中未找到图像文件")
        first_image = read_image(os.path.join(image_dir, image_files[0]))
        self.image_size = (first_image.shape[1], first_image.shape[0])

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def single_camera_calibration(self):
        points_world, points_pixel = self._step0_detect_corners()
        homographies, intrinsic, truncated = self._step1_solve_H_A_Ks(points_world, points_pixel)
        magnitudes = self._step2_solve_r31_r32_magnitudes(truncated)
        signs = self._step3_determine_r31_r32_signs(points_world, points_pixel)
        rotations = self._step4_construct_rotation_matrices(truncated, magnitudes, signs)
        final_params = self._step5_global_optimization(
            points_world, points_pixel, homographies, intrinsic, truncated, rotations
        )
        self.final_params = final_params
        return final_params

    def _step0_detect_corners(self, image_path=None):
        """从目录读取图像，返回每张图的世界点和像素点。"""
        if image_path is None:
            image_path = self.image_dir
        self.pattern_board = pattern_board_from_holder(self.param_holder)
        pat_size = self.param_holder.get_board_dict()["dot_array_shape"]
        image_files = list_calibration_images(image_path)
        if not image_files:
            raise ValueError(f"在目录 {image_path} 中未找到图像文件")

        self._print("\n🔍 使用github_halcondetector检测标定板...")
        self._print(f"   标定板尺寸: {pat_size[0]}x{pat_size[1]}")
        self._print(f"   找到 {len(image_files)} 张图像")

        points_world = []
        points_pixel = []
        for index, image_file in enumerate(image_files):
            try:
                image = read_image(image_file)
            except Exception as exc:
                self._print(f"⚠️  无法读取图像: {image_file} - {exc}")
                continue
            success, centers, detection_method = detect_calibration_image(image, pat_size)
            if success:
                points_pixel.append(centers)
                points_world.append(self.pattern_board.generate_world_points_grid())
                self._print(
                    f"   ✓ 图像 {index + 1}/{len(image_files)}: "
                    f"检测到 {len(centers)} 个标记点 [{detection_method}]"
                )
            else:
                self._print(f"   ✗ 图像 {index + 1}/{len(image_files)}: 检测失败")

        if len(points_pixel) == 0:
            raise ValueError("所有图像的角点检测都失败了！")
        self._print(f"   📊 成功检测 {len(points_pixel)}/{len(image_files)} 张图像")
        return points_world, points_pixel

    def _step1_solve_H_A_Ks(self, points_world, points_pixel):
        return solve_h_a_ks(points_world, points_pixel)

    def _step2_solve_r31_r32_magnitudes(self, truncated_extrinsics):
        return solve_r31_r32_magnitudes(truncated_extrinsics)

    def _step3_determine_r31_r32_signs(self, points_world, points_pixel):
        return determine_r31_r32_signs(points_world, points_pixel, self.image_size, self._print)

    def _step4_construct_rotation_matrices(self, truncated_extrinsics, magnitudes, signs):
        return construct_rotation_matrices(truncated_extrinsics, magnitudes, signs)

    def _step5_global_optimization(
        self, points_world, points_pixel, homographies, intrinsic, truncated, rotations
    ):
        optimized = global_optimize(
            points_world,
            points_pixel,
            homographies,
            intrinsic,
            truncated,
            rotations,
            self.verbose,
            self._print,
        )
        alpha, beta, gamma, k1, k2, rotations_opt, tvecs = optimized
        self.param_holder.set_intrinsic(
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            k1=float(k1),
            k2=float(k2),
        )
        self.param_holder.set_extrinsic(
            R=rotations_opt[0],
            T=np.array([tvecs[0][0], tvecs[0][1], 0]),
        )
        self._print(f"   📐 优化后内参: alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")
        self._print(f"   🔍 优化后畸变: k1={k1:.6e}, k2={k2:.6e}")
        return optimized

    def known_intrinsic_calibration_single_image(self, image_path: str):
        if self.final_params is None:
            raise ValueError("final_params未确定，请先确定单相机参数，使用single_camera_calibration方法")
        image = read_image(image_path)
        pat_size = self.param_holder.get_board_dict()["dot_array_shape"]
        success, centers, _method = detect_calibration_image(image, pat_size)
        if not success:
            raise ValueError(f"角点检测失败: {image_path}")
        alpha, beta, gamma = self.final_params[0], self.final_params[1], self.final_params[2]
        return solve_pose_with_known_intrinsics(
            self.pattern_board.generate_world_points_grid(),
            centers,
            alpha,
            beta,
            gamma,
            self.image_size,
            image_path,
            self._print,
        )

    def save_camera_params_to_json(self, output_path: str, image_index: int = 0):
        if self.final_params is None:
            raise ValueError("final_params未确定，请先执行single_camera_calibration")
        alpha, beta, gamma, k1, k2, rotations, translations = self.final_params
        if image_index >= len(rotations):
            self._print(f"⚠️  警告：image_index={image_index}超出范围，使用第一张图像的外参")
            image_index = 0
        output = build_camera_params_dict(
            alpha, beta, gamma, k1, k2, rotations[image_index], translations[image_index]
        )
        save_json(output, output_path)
        self._print(f"✅ 相机参数已保存到: {output_path}")
        return output

    def save_ck_matrix_json(self, rvec, tvec, rotation, output_path="CKmatrix.json"):
        ck_matrix, data = build_ck_matrix_dict(
            self.param_holder.alpha,
            self.param_holder.beta,
            self.param_holder.gamma,
            self.param_holder.k1,
            self.param_holder.k2,
            rvec,
            tvec,
            rotation,
        )
        save_json(data, output_path)
        self._print(f"   ✅ CK矩阵已保存到: {output_path}")
        return ck_matrix
