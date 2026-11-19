"""标定板几何：阵列尺寸、间距和世界坐标。"""
import numpy as np


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


