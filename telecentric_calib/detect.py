"""标定板圆点检测。阈值重试只在这里发生。"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

from .halcon_detector import FindCirclesGridPattern, FindHalconCalibBoard
from .io import read_image


def detect_pattern(
    image: np.ndarray,
    pat_size: Tuple[int, int],
    auto_threshold: bool = True,
) -> Tuple[bool, Optional[List], Optional[List], Optional[str]]:
    """
    检测标定板图案。

    先用圆形网格，失败后再用 Halcon 标定板。auto_threshold 为真时，
    自动阈值失败会从 10 到 240 每 10 级再试一次。

    返回:
        (success, centers, ellipses, method)
    """
    success, centers, ellipses = FindCirclesGridPattern(image, pat_size)
    detection_method = None

    if not success and auto_threshold:
        for thresh in range(10, 250, 10):
            success, centers, ellipses = FindCirclesGridPattern(image, pat_size, thresh)
            if success:
                detection_method = f"FindCirclesGridPattern (阈值={thresh})"
                break
    elif success:
        detection_method = "FindCirclesGridPattern (自动阈值)"

    if not success:
        success, centers, ellipses = FindHalconCalibBoard(image, pat_size)
        if not success and auto_threshold:
            for thresh in range(10, 250, 10):
                success, centers, ellipses = FindHalconCalibBoard(image, pat_size, thresh)
                if success:
                    detection_method = f"FindHalconCalibBoard (阈值={thresh})"
                    break
        elif success:
            detection_method = "FindHalconCalibBoard (自动阈值)"

    return success, centers, ellipses, detection_method


def detect_pattern_from_path(
    image_path: str,
    pat_size: Tuple[int, int],
    auto_threshold: bool = True,
) -> Tuple[bool, Optional[List], Optional[List], Optional[str]]:
    """从文件路径检测标定板图案。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return detect_pattern(image, pat_size, auto_threshold)


def detect_calibration_image(image: np.ndarray, pat_size: Tuple[int, int]):
    """检测一张标定图，成功时返回像素点数组和检测方式。"""
    success, centers, _ellipses, detection_method = detect_pattern(image, pat_size)
    if not success:
        return False, None, detection_method
    centers_array = np.array(centers, dtype=np.float32).reshape(-1, 2)
    return True, centers_array, detection_method
