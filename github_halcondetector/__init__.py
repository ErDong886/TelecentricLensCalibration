"""
find_calib_pattern - Python版本
用于检测和识别图像中的标定板圆形标记点

Author: Converted from C++ version by Neucrede
"""

# 支持多种导入方式
try:
    from .find_calib_pattern import FindCirclesGridPattern, FindHalconCalibBoard
except ImportError:
    from find_calib_pattern import FindCirclesGridPattern, FindHalconCalibBoard

__version__ = "1.0.0"
__all__ = ['FindCirclesGridPattern', 'FindHalconCalibBoard']



