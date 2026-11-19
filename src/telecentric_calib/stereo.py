"""双目标定、去畸变和三维重建。"""
import json
import os

import cv2
import numpy as np

from .detect.fallback import find_centers
from .io import read_image
from .mono import MonoCalibration


class DualCameraCalibrator:
    def __init__(self, left_camera: MonoCalibration, right_camera: MonoCalibration, binocular_image_dir: str, 
                 left_image_name:str = "0_left.bmp", right_image_name:str = "0_right.bmp", verbose:bool = True):
        """
        初始化双目标定器
        
        参数:
            left_camera: 左相机标定器
            right_camera: 右相机标定器
            binocular_image_dir: 双目图像目录
            left_image_name: 左相机图像文件名（默认: "0_left.bmp"）
            right_image_name: 右相机图像文件名（默认: "0_right.bmp"）
            verbose: 是否显示详细输出（默认True）
        """
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.binocular_image_dir = binocular_image_dir
        self.left_image_name = left_image_name
        self.right_image_name = right_image_name
        self.verbose = verbose
    
    def _print(self, *args, **kwargs):
        """条件输出方法，仅在verbose=True时输出"""
        if self.verbose:
            print(*args, **kwargs)
        
    def undistort_points(self, points_distorted, k1, k2, max_iter=20, tolerance=1e-6):
        """
        对畸变的像素坐标进行去畸变处理
        
        畸变模型: p_distorted = p_ideal * (1 + k1*r^2 + k2*r^4)
        其中 r^2 = u_ideal^2 + v_ideal^2
        
        参数:
            points_distorted: 带畸变的像素坐标 (N, 2)
            k1, k2: 畸变系数
            max_iter: 最大迭代次数
            tolerance: 收敛容差
        
        返回:
            points_ideal: 去畸变后的理想像素坐标 (N, 2)
        """
        points_ideal = points_distorted.copy()
        
        for iteration in range(max_iter):
            r_sq = points_ideal[:, 0]**2 + points_ideal[:, 1]**2
            distortion_factor = 1 + k1 * r_sq + k2 * r_sq**2
            
            # 根据当前理想坐标计算应该产生的畸变坐标
            points_predicted = points_ideal * distortion_factor[:, np.newaxis]
            
            # 计算误差
            error = points_distorted - points_predicted
            
            # 更新理想坐标
            points_ideal = points_ideal + error / distortion_factor[:, np.newaxis]
            
            # 检查收敛
            if np.max(np.abs(error)) < tolerance:
                break
        
        return points_ideal
    
    def dual_camera_calibration(self):
        # 利用已经标定好的单相机参数，标定双相机参数
        # 读取左右相机的共同图像（使用配置的文件名）
        left_image_path = os.path.join(self.binocular_image_dir, self.left_image_name)
        right_image_path = os.path.join(self.binocular_image_dir, self.right_image_name)
        
        # 检测左右图像的角点
        self._print("\n===== 双目标定：检测对应点 =====")
        left_image = read_image(left_image_path)
        right_image = read_image(right_image_path)
        
        board_params = self.left_camera.param_holder.get_board_dict()
        pat_size = board_params['dot_array_shape']
        
        success_left, centers_left, _, _ = find_centers(left_image, pat_size)
        success_right, centers_right, _, _ = find_centers(right_image, pat_size)
        if not success_left or not success_right:
            raise ValueError("角点检测失败！")
        
        points_pixel_left = np.array(centers_left, dtype=np.float32).reshape(-1, 2)
        points_pixel_right = np.array(centers_right, dtype=np.float32).reshape(-1, 2)
        
        self._print(f"检测到 {len(points_pixel_left)} 对对应点")
        
        # 获取外参
        left_rvec, left_tvec, left_R = self.left_camera.known_intrinsic_calibration_single_image(left_image_path)
        right_rvec, right_tvec, right_R = self.right_camera.known_intrinsic_calibration_single_image(right_image_path)
        
        ## 利用左右相机的内外参矩阵，得到最终M矩阵
        Mat_intrinsic_left = np.array([
            [self.left_camera.param_holder.alpha, self.left_camera.param_holder.gamma, 0, 0],
            [0, self.left_camera.param_holder.beta, 0, 0],
            [0, 0, 0, 1]
        ])
        Mat_intrinsic_right = np.array([
            [self.right_camera.param_holder.alpha, self.right_camera.param_holder.gamma, 0, 0],
            [0, self.right_camera.param_holder.beta, 0, 0],
            [0, 0, 0, 1]
        ])
        Mat_extrinsic_left = np.array([
            [left_R[0, 0], left_R[0, 1], left_R[0, 2], left_tvec[0]],
            [left_R[1, 0], left_R[1, 1], left_R[1, 2], left_tvec[1]],
            [left_R[2, 0], left_R[2, 1], left_R[2, 2], 0],
            [0, 0, 0, 1]
        ])
        Mat_extrinsic_right = np.array([
            [right_R[0, 0], right_R[0, 1], right_R[0, 2], right_tvec[0]],
            [right_R[1, 0], right_R[1, 1], right_R[1, 2], right_tvec[1]],
            [right_R[2, 0], right_R[2, 1], right_R[2, 2], 0],
            [0, 0, 0, 1]
        ])
        
        Mat_left = Mat_intrinsic_left @ Mat_extrinsic_left
        Mat_right = Mat_intrinsic_right @ Mat_extrinsic_right

        self._print(f"\n左相机M矩阵:\n{Mat_left}")
        self._print(f"\n右相机M矩阵:\n{Mat_right}")
        
        # 去畸变处理
        self._print("\n===== 去畸变处理 =====")
        k1_left = self.left_camera.param_holder.k1
        k2_left = self.left_camera.param_holder.k2
        k1_right = self.right_camera.param_holder.k1
        k2_right = self.right_camera.param_holder.k2
        
        self._print(f"左相机畸变系数: k1={k1_left:.6e}, k2={k2_left:.6e}")
        self._print(f"右相机畸变系数: k1={k1_right:.6e}, k2={k2_right:.6e}")
        
        points_ideal_left = self.undistort_points(points_pixel_left, k1_left, k2_left)
        points_ideal_right = self.undistort_points(points_pixel_right, k1_right, k2_right)
        
        # 获取理论世界坐标（用于对比）
        points_world_theory = self.left_camera.pattern_board.generate_world_points_grid()
        
        # 从M矩阵中提取前两行（对应u和v方程）
        M_left = Mat_left[:2, :]  # 2x4
        M_right = Mat_right[:2, :]  # 2x4
        
        # ========== 对比实验1：不带畸变校正的重建 ==========
        self._print("\n" + "="*60)
        self._print("对比实验1：使用原始像素坐标（不带畸变校正）")
        self._print("="*60)
        
        reconstructed_points_distorted = []
        
        for i in range(len(points_pixel_left)):
            p = points_pixel_left[i]  # 带畸变的原始像素坐标
            p_prime = points_pixel_right[i]
            
            # 构建线性方程组
            A = np.vstack([
                M_left[:, :3],
                M_right[:, :3]
            ])
            
            b = np.hstack([
                p - M_left[:, 3],
                p_prime - M_right[:, 3]
            ])
            
            P, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            reconstructed_points_distorted.append(P)
        
        reconstructed_points_distorted = np.array(reconstructed_points_distorted)
        errors_distorted = reconstructed_points_distorted - points_world_theory
        
        self._print("\n【不带畸变校正的重建精度】")
        self._print(f"X方向 - 均值: {np.mean(errors_distorted[:, 0]):.4f} mm, 标准差: {np.std(errors_distorted[:, 0]):.4f} mm, 最大值: {np.max(np.abs(errors_distorted[:, 0])):.4f} mm")
        self._print(f"Y方向 - 均值: {np.mean(errors_distorted[:, 1]):.4f} mm, 标准差: {np.std(errors_distorted[:, 1]):.4f} mm, 最大值: {np.max(np.abs(errors_distorted[:, 1])):.4f} mm")
        self._print(f"Z方向 - 均值: {np.mean(errors_distorted[:, 2]):.4f} mm, 标准差: {np.std(errors_distorted[:, 2]):.4f} mm, 最大值: {np.max(np.abs(errors_distorted[:, 2])):.4f} mm")
        self._print(f"整体RMS误差: {np.sqrt(np.mean(errors_distorted**2)):.4f} mm")
        
        # ========== 对比实验2：带畸变校正的重建 ==========
        self._print("\n" + "="*60)
        self._print("对比实验2：使用去畸变后的像素坐标（带畸变校正）")
        self._print("="*60)
        
        reconstructed_points_undistorted = []
        
        for i in range(len(points_ideal_left)):
            p = points_ideal_left[i]  # 去畸变后的理想像素坐标
            p_prime = points_ideal_right[i]
            
            # 构建线性方程组
            A = np.vstack([
                M_left[:, :3],
                M_right[:, :3]
            ])
            
            b = np.hstack([
                p - M_left[:, 3],
                p_prime - M_right[:, 3]
            ])
            
            P, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            reconstructed_points_undistorted.append(P)
        
        reconstructed_points_undistorted = np.array(reconstructed_points_undistorted)
        errors_undistorted = reconstructed_points_undistorted - points_world_theory
        
        self._print("\n【带畸变校正的重建精度】")
        self._print(f"X方向 - 均值: {np.mean(errors_undistorted[:, 0]):.4f} mm, 标准差: {np.std(errors_undistorted[:, 0]):.4f} mm, 最大值: {np.max(np.abs(errors_undistorted[:, 0])):.4f} mm")
        self._print(f"Y方向 - 均值: {np.mean(errors_undistorted[:, 1]):.4f} mm, 标准差: {np.std(errors_undistorted[:, 1]):.4f} mm, 最大值: {np.max(np.abs(errors_undistorted[:, 1])):.4f} mm")
        self._print(f"Z方向 - 均值: {np.mean(errors_undistorted[:, 2]):.4f} mm, 标准差: {np.std(errors_undistorted[:, 2]):.4f} mm, 最大值: {np.max(np.abs(errors_undistorted[:, 2])):.4f} mm")
        self._print(f"整体RMS误差: {np.sqrt(np.mean(errors_undistorted**2)):.4f} mm")
        
        # ========== 对比总结 ==========
        self._print("\n" + "="*60)
        self._print("📊 畸变校正效果对比总结")
        self._print("="*60)
        
        rms_distorted = np.sqrt(np.mean(errors_distorted**2))
        rms_undistorted = np.sqrt(np.mean(errors_undistorted**2))
        
        self._print(f"\n不带畸变校正 RMS误差: {rms_distorted:.4f} mm")
        self._print(f"带畸变校正 RMS误差:   {rms_undistorted:.4f} mm")
        if rms_distorted == 0:
            self._print("未校正 RMS 为 0，不计算提升比例")
        else:
            improvement = (rms_distorted - rms_undistorted) / rms_distorted * 100
            self._print(f"精度提升: {improvement:.2f}%")
        
        return Mat_left, Mat_right, reconstructed_points_undistorted, points_world_theory, {
            'distorted': reconstructed_points_distorted,
            'undistorted': reconstructed_points_undistorted,
            'errors_distorted': errors_distorted,
            'errors_undistorted': errors_undistorted,
            'points_pixel_left': points_pixel_left,
            'points_pixel_right': points_pixel_right
        }
    
    def save_fundamental_matrix_json(self, points_left, points_right, output_path="fundamentalMatrix.json"):
        """
        计算并保存基础矩阵到JSON文件
        
        参数:
            points_left: 左相机像素坐标点
            points_right: 右相机像素坐标点
            output_path: 输出文件路径
        """
        # 使用RANSAC方法计算基础矩阵
        F_matrix, mask = cv2.findFundamentalMat(
            points_left, 
            points_right, 
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99
        )
        
        if F_matrix is None:
            self._print("   ⚠️  基础矩阵计算失败")
            return None
        
        # 统计内点数量
        inliers = np.sum(mask)
        total_points = len(points_left)
        
        # 准备JSON数据
        data = {
            "fundamental_matrix": F_matrix.tolist(),
            "description": "使用多组同名点计算得到的基础矩阵",
            "total_points": int(total_points),
            "inliers": int(inliers),
            "method": "RANSAC"
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        self._print(f"   ✅ 基础矩阵已保存到: {output_path}")
        self._print(f"   总点数: {total_points}, 内点数: {inliers} ({inliers/total_points*100:.1f}%)")
        
        return F_matrix
    
    def save_reconstruction_errors_json(self, points_theory, points_reconstructed_distorted, 
                                       points_reconstructed_undistorted, output_path="reconstruction_errors.json"):
        """
        保存每个点的重建误差统计到JSON文件
        
        参数:
            points_theory: 理论标记点坐标 (N, 3)
            points_reconstructed_distorted: 不带畸变校正的重建点 (N, 3)
            points_reconstructed_undistorted: 带畸变校正的重建点 (N, 3)
            output_path: 输出文件路径
        """
        # 计算误差
        errors_distorted = points_reconstructed_distorted - points_theory
        errors_undistorted = points_reconstructed_undistorted - points_theory
        
        # 构建每个点的详细信息
        points_details = []
        for i in range(len(points_theory)):
            point_info = {
                "point_index": i,
                "theory": {
                    "X": float(points_theory[i, 0]),
                    "Y": float(points_theory[i, 1]),
                    "Z": float(points_theory[i, 2])
                },
                "reconstructed_without_distortion_correction": {
                    "X": float(points_reconstructed_distorted[i, 0]),
                    "Y": float(points_reconstructed_distorted[i, 1]),
                    "Z": float(points_reconstructed_distorted[i, 2])
                },
                "reconstructed_with_distortion_correction": {
                    "X": float(points_reconstructed_undistorted[i, 0]),
                    "Y": float(points_reconstructed_undistorted[i, 1]),
                    "Z": float(points_reconstructed_undistorted[i, 2])
                },
                "error_without_correction": {
                    "dX": float(errors_distorted[i, 0]),
                    "dY": float(errors_distorted[i, 1]),
                    "dZ": float(errors_distorted[i, 2]),
                    "norm": float(np.linalg.norm(errors_distorted[i]))
                },
                "error_with_correction": {
                    "dX": float(errors_undistorted[i, 0]),
                    "dY": float(errors_undistorted[i, 1]),
                    "dZ": float(errors_undistorted[i, 2]),
                    "norm": float(np.linalg.norm(errors_undistorted[i]))
                }
            }
            points_details.append(point_info)
        
        # 计算统计信息
        statistics = {
            "without_distortion_correction": {
                "total_points": len(points_theory),
                "mean_error": {
                    "X": float(np.mean(errors_distorted[:, 0])),
                    "Y": float(np.mean(errors_distorted[:, 1])),
                    "Z": float(np.mean(errors_distorted[:, 2]))
                },
                "std_error": {
                    "X": float(np.std(errors_distorted[:, 0])),
                    "Y": float(np.std(errors_distorted[:, 1])),
                    "Z": float(np.std(errors_distorted[:, 2]))
                },
                "max_abs_error": {
                    "X": float(np.max(np.abs(errors_distorted[:, 0]))),
                    "Y": float(np.max(np.abs(errors_distorted[:, 1]))),
                    "Z": float(np.max(np.abs(errors_distorted[:, 2])))
                },
                "rms_error": float(np.sqrt(np.mean(errors_distorted**2)))
            },
            "with_distortion_correction": {
                "total_points": len(points_theory),
                "mean_error": {
                    "X": float(np.mean(errors_undistorted[:, 0])),
                    "Y": float(np.mean(errors_undistorted[:, 1])),
                    "Z": float(np.mean(errors_undistorted[:, 2]))
                },
                "std_error": {
                    "X": float(np.std(errors_undistorted[:, 0])),
                    "Y": float(np.std(errors_undistorted[:, 1])),
                    "Z": float(np.std(errors_undistorted[:, 2]))
                },
                "max_abs_error": {
                    "X": float(np.max(np.abs(errors_undistorted[:, 0]))),
                    "Y": float(np.max(np.abs(errors_undistorted[:, 1]))),
                    "Z": float(np.max(np.abs(errors_undistorted[:, 2])))
                },
                "rms_error": float(np.sqrt(np.mean(errors_undistorted**2)))
            }
        }
        
        # 计算改善率
        rms_distorted = statistics["without_distortion_correction"]["rms_error"]
        rms_undistorted = statistics["with_distortion_correction"]["rms_error"]
        improvement = (rms_distorted - rms_undistorted) / rms_distorted * 100 if rms_distorted != 0 else 0
        
        # 构建完整输出
        output_data = {
            "description": "三维重建误差统计（理论标记点 vs 重建点）",
            "unit": "mm",
            "statistics": statistics,
            "improvement_percentage": float(improvement),
            "points_details": points_details
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        self._print(f"   ✅ 重建误差统计已保存到: {output_path}")
        self._print(f"   总点数: {len(points_theory)}")
        self._print(f"   不带校正 RMS误差: {rms_distorted:.4f} mm")
        self._print(f"   带校正 RMS误差: {rms_undistorted:.4f} mm")
        if rms_distorted == 0:
            self._print("   未校正 RMS 为 0，不计算提升比例")
        else:
            self._print(f"   精度提升: {improvement:.2f}%")
        
        return output_data

