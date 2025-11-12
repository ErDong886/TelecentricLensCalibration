"""
检测器模块
使用github_halcondetector进行标定板检测
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# 添加github_halcondetector到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
halcon_path = os.path.join(parent_dir, 'github_halcondetector')
if halcon_path not in sys.path:
    sys.path.insert(0, halcon_path)

try:
    from github_halcondetector import FindHalconCalibBoard, FindCirclesGridPattern
except ImportError:
    # 如果直接导入失败，尝试相对导入
    try:
        from find_calib_pattern import FindHalconCalibBoard, FindCirclesGridPattern
    except ImportError:
        raise ImportError("无法导入github_halcondetector模块，请确保该模块在Python路径中")


class PatternBoard:
    """标定板信息类"""
    
    def __init__(self, pattern_type: int, shape: tuple, distance: float, pixel_size: tuple, radius: float = None):
        """
        :param pattern_type: PatternType标定图案样式.
                              0: 棋盘格，1：halcon标定板，2：...
        :param shape: (x, y) 控制点形状
        :param distance: 控制点间距(单位：mm)
        :param pixel_size: (x, y) 显示器像素尺寸，用于将真实尺寸转换为像素尺寸(单位：um)
        :param radius: 对于halcon标定板，圆点半径(单位：mm)
        """
        # 参数类型和范围检查
        if not isinstance(pattern_type, int) or pattern_type not in [0, 1]:
            raise ValueError("pattern_type must be an integer, either 0 (棋盘格) or 1 (halcon标定板).")

        if not isinstance(shape, tuple) or len(shape) != 2 or not all(
                isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError("shape must be a tuple of two positive integers (x, y).")

        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError("distance must be a positive number (greater than 0).")

        if not isinstance(pixel_size, tuple) or len(pixel_size) != 2 or not all(
                isinstance(dim, (int, float)) and dim > 0 for dim in pixel_size):
            raise ValueError("pixel_size must be a tuple of two positive numbers (x, y).")

        # 如果pattern_type是1，radius必须提供
        if pattern_type == 1 and radius is None:
            raise ValueError("For pattern_type 1 (halcon标定板), radius must be provided.")

        self.pattern_type = pattern_type  # 标定图案样式
        self.shape = shape  # 控制点形状
        self.distance = distance  # 控制点间距(单位：mm)
        self.pixel_size = pixel_size  # 显示器像素尺寸，用于将真实尺寸转换为像素尺寸(单位：um)
        self.radius = radius  # 圆点半径 (单位：mm)

    def generate_world_points_grid(self):
        """
        依据标定板的形状与点间距，生成世界坐标系下的角点网格（Z=0）。

        返回:
            np.ndarray: 形状为 (num_points, 3) 的世界坐标数组，单位为 mm。
        """
        w, h = self.shape
        cp_int = np.zeros((w * h, 3), np.float32)
        # 再次交换x和y坐标的生成顺序，恢复原始的坐标分配
        for i in range(h):
            for j in range(w):
                cp_int[i * w + j] = [i, j, 0]
        return cp_int * self.distance


def detect_pattern(image: np.ndarray, pat_size: Tuple[int, int], 
                   auto_threshold: bool = True) -> Tuple[bool, List, Optional[List]]:
    """
    检测标定板图案
    
    参数:
        image: 输入图像 (BGR格式)
        pat_size: 标定板尺寸 (rows, cols)
        auto_threshold: 是否自动尝试不同阈值
    
    返回:
        (success, centers, ellipses)
        success: 是否检测成功
        centers: 检测到的中心点列表
        ellipses: 椭圆信息列表（如果有）
    """
    # 方法1: 尝试通用圆形网格检测
    success, centers, ellipses = FindCirclesGridPattern(image, pat_size)
    detection_method = None
    
    if not success and auto_threshold:
        # 尝试不同的阈值
        for thresh in range(10, 250, 10):
            success, centers, ellipses = FindCirclesGridPattern(image, pat_size, thresh)
            if success:
                detection_method = f"FindCirclesGridPattern (阈值={thresh})"
                break
    else:
        detection_method = "FindCirclesGridPattern (自动阈值)"
    
    # 如果第一种方法失败，尝试Halcon方法
    if not success:
        success, centers, ellipses = FindHalconCalibBoard(image, pat_size)
        
        if not success and auto_threshold:
            for thresh in range(10, 250, 10):
                success, centers, ellipses = FindHalconCalibBoard(image, pat_size, thresh)
                if success:
                    detection_method = f"FindHalconCalibBoard (阈值={thresh})"
                    break
        else:
            detection_method = "FindHalconCalibBoard (自动阈值)"
    
    return success, centers, ellipses


def detect_pattern_from_path(image_path: str, pat_size: Tuple[int, int], 
                             auto_threshold: bool = True) -> Tuple[bool, List, Optional[List]]:
    """
    从文件路径检测标定板图案
    
    参数:
        image_path: 图像文件路径
        pat_size: 标定板尺寸 (rows, cols)
        auto_threshold: 是否自动尝试不同阈值
    
    返回:
        (success, centers, ellipses)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    return detect_pattern(image, pat_size, auto_threshold)

