"""
参数管理模块
用于存储和管理相机标定参数
"""
import numpy as np
from typing import Optional, Tuple


class CalibParamHolder:
    """标定参数容器类，用于存储相机和标定板参数"""
    
    def __init__(
        self,
        # 相机内参
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        # 相机外参
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        # 标定板参数
        dot_array_shape: Optional[Tuple[int, int]] = None,
        dot_diameter: Optional[float] = None,
        dot_spacing: Optional[float] = None
    ):
        """
        初始化标定参数
        
        Args:
            alpha: 内参alpha
            beta: 内参beta
            gamma: 内参gamma
            k1: 畸变系数k1
            k2: 畸变系数k2
            R: 旋转矩阵 (3x3)
            T: 平移向量 (3x1)
            dot_array_shape: 圆点阵列形状 (rows, cols)
            dot_diameter: 圆点直径
            dot_spacing: 圆点中心间距
        """
        # 相机内参
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k1 = k1
        self.k2 = k2
        
        # 相机外参
        self.R = np.eye(3) if R is None else np.asarray(R)
        self.T = np.zeros((3, 1)) if T is None else np.asarray(T).reshape(3, 1)
        
        # 标定板参数
        self.dot_array_shape = dot_array_shape
        self.dot_diameter = dot_diameter
        self.dot_spacing = dot_spacing
    
    def get_intrinsic_dict(self) -> dict:
        """获取内参字典"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'k1': self.k1,
            'k2': self.k2
        }
    
    def get_extrinsic_dict(self) -> dict:
        """获取外参字典"""
        return {
            'R': self.R,
            'T': self.T
        }
    
    def get_board_dict(self) -> dict:
        """获取标定板参数字典"""
        return {
            'dot_array_shape': self.dot_array_shape,
            'dot_diameter': self.dot_diameter,
            'dot_spacing': self.dot_spacing
        }
    
    def set_intrinsic(self, alpha=None, beta=None, gamma=None, k1=None, k2=None):
        """批量设置内参"""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if k1 is not None:
            self.k1 = k1
        if k2 is not None:
            self.k2 = k2
    
    def set_extrinsic(self, R=None, T=None):
        """批量设置外参"""
        if R is not None:
            self.R = np.asarray(R)
        if T is not None:
            self.T = np.asarray(T).reshape(3, 1)
    
    def set_board(self, dot_array_shape=None, dot_diameter=None, dot_spacing=None):
        """批量设置标定板参数"""
        if dot_array_shape is not None:
            self.dot_array_shape = dot_array_shape
        if dot_diameter is not None:
            self.dot_diameter = dot_diameter
        if dot_spacing is not None:
            self.dot_spacing = dot_spacing
    
    def __repr__(self):
        return (
            f"CalibParamHolder(\n"
            f"  内参: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, k1={self.k1}, k2={self.k2}\n"
            f"  外参: R.shape={self.R.shape}, T.shape={self.T.shape}\n"
            f"  标定板: shape={self.dot_array_shape}, diameter={self.dot_diameter}, spacing={self.dot_spacing}\n"
            f")"
        )


class PatternBoard:
    """标定板几何：控制点形状、间距，以及 Z=0 的世界坐标网格。"""

    def __init__(self, pattern_type: int, shape: tuple, distance: float, pixel_size: tuple, radius: float = None):
        """
        :param pattern_type: 0 为棋盘格，1 为 halcon 标定板
        :param shape: (x, y) 控制点形状
        :param distance: 控制点间距，单位 mm
        :param pixel_size: (x, y) 像素尺寸，单位 um
        :param radius: halcon 圆点半径，单位 mm
        """
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

        if pattern_type == 1 and radius is None:
            raise ValueError("For pattern_type 1 (halcon标定板), radius must be provided.")

        self.pattern_type = pattern_type
        self.shape = shape
        self.distance = distance
        self.pixel_size = pixel_size
        self.radius = radius

    def generate_world_points_grid(self):
        """生成世界坐标系下的控制点网格（Z=0），单位为 mm。"""
        width, height = self.shape
        points = np.zeros((width * height, 3), np.float32)
        for row in range(height):
            for col in range(width):
                points[row * width + col] = [row, col, 0]
        return points * self.distance


def pattern_board_from_holder(param_holder: CalibParamHolder) -> PatternBoard:
    """用参数容器里的标定板设置构造 PatternBoard。"""
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

