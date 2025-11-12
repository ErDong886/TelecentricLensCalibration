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

