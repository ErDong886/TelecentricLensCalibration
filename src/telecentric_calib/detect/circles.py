"""标定板圆点检测。

移植自 Neucrede/find_calib_pattern，BSD-2-Clause License。
原作者 Neucrede <neucrede@sina.com>。
"""

import numpy as np
import cv2

from .ellipse import fit_ellipse_subpixel
from .geometry import (
    _find_four_corners,
    _hierarchical_clustering,
    _sort_ellipses_and_center_points,
)


def FindCirclesGridPattern(img, pat_size=(7, 7), thresh=-1, inverse_thresh=False,
                           sub_pixel=True, mask=None, corner_points_hint=None):
    """
    查找圆网格标定板模式
    
    参数:
        img: 输入图像
        pat_size: 标定板尺寸 (width, height)
        thresh: 二值化阈值，-1表示自动
        inverse_thresh: 是否反向二值化
        sub_pixel: 是否使用亚像素精度
        mask: 掩码图像
        corner_points_hint: 角点提示 [(x,y), ...]
    
    返回:
        success: 是否成功
        sorted_center_points: 排序后的中心点列表
        sorted_ellipses: 排序后的椭圆列表 (可选)
    """
    if img is None or img.size == 0:
        raise ValueError("img is empty")
    
    if pat_size[0] < 2 or pat_size[1] < 2:
        raise ValueError("Pattern size can't be smaller than 2x2")
    
    # 转换为灰度图
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    # 高斯模糊
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    
    # 应用掩码
    if mask is not None:
        if mask.dtype != np.uint8:
            raise ValueError("Bad mask data type. Must be uint8")
        img_gray = cv2.bitwise_and(img_gray, mask)
    
    # 二值化
    if thresh < 0:
        thresh_type = cv2.THRESH_BINARY_INV if inverse_thresh else cv2.THRESH_BINARY
        _, img_mono = cv2.threshold(img_gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        thresh_type = cv2.THRESH_BINARY_INV if inverse_thresh else cv2.THRESH_BINARY
        _, img_mono = cv2.threshold(img_gray, thresh, 255, thresh_type)
    
    # 提取轮廓
    contours, hierarchy = cv2.findContours(img_mono, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    total = pat_size[0] * pat_size[1]
    
    if len(contours) < total:
        return False, None, None
    
    # 计算图像梯度
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    
    # 拟合椭圆
    ellipses = []
    for contour in contours:
        moments = cv2.moments(contour)
        area = moments['m00']
        
        # 按面积过滤
        area_thresh = img.shape[0] * img.shape[1] / (pat_size[0] * pat_size[1])
        if area > area_thresh:
            continue
        
        # 按圆度过滤
        perim = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (perim * perim + 1.0)
        if circularity < 0.75:
            continue
        
        # 按凸性过滤
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if area / (hull_area + 1.0) < 0.9:
            continue
        
        # 拟合椭圆
        if sub_pixel:
            success, ellipse = fit_ellipse_subpixel(grad_x, grad_y, contour.reshape(-1, 2))
            if not success:
                continue
        elif len(contour) >= 6:
            ellipse = cv2.fitEllipseDirect(contour)
        else:
            continue
        
        bound_rect = cv2.boundingRect(contour)
        if bound_rect[2] * bound_rect[3] >= 64:
            ellipses.append(ellipse)
    
    if len(ellipses) < total:
        return False, None, None
    
    # 过滤过小的blob
    max_ellipse_area = max(np.pi * e[1][0] * e[1][1] / 4 for e in ellipses)
    area0 = 0.5 * max_ellipse_area
    good_ellipses = [e for e in ellipses if np.pi * e[1][0] * e[1][1] / 4 >= area0]
    ellipses = good_ellipses
    
    num_ellipses = len(ellipses)
    
    # 找到最大聚类
    points = [e[0] for e in ellipses]
    blob_indices = _hierarchical_clustering(points, pat_size)
    
    if blob_indices is None:
        return False, None, None
    
    # 计算长短轴的均值和标准差
    minor_lengths = []
    major_lengths = []
    
    for idx in blob_indices:
        ellipse = ellipses[idx]
        minor_length = min(ellipse[1][0], ellipse[1][1])
        major_length = max(ellipse[1][0], ellipse[1][1])
        minor_lengths.append(minor_length)
        major_lengths.append(major_length)
    
    mean_minor = np.mean(minor_lengths)
    mean_major = np.mean(major_lengths)
    stddev_minor = np.std(minor_lengths)
    stddev_major = np.std(major_lengths)
    
    # 有效长度范围
    low_min_len = mean_minor - max(0.5 * mean_minor, 3.0 * stddev_minor)
    upp_min_len = mean_minor + max(0.5 * mean_minor, 3.0 * stddev_minor)
    low_maj_len = mean_major - max(0.5 * mean_major, 3.0 * stddev_major)
    upp_maj_len = mean_major + max(0.5 * mean_major, 3.0 * stddev_major)
    
    # 过滤椭圆
    center_points = []
    ellipses_filtered = []
    
    for idx in blob_indices:
        ellipse = ellipses[idx]
        minor_length = min(ellipse[1][0], ellipse[1][1])
        major_length = max(ellipse[1][0], ellipse[1][1])
        
        if not (low_min_len <= minor_length <= upp_min_len):
            continue
        if not (low_maj_len <= major_length <= upp_maj_len):
            continue
        
        ellipses_filtered.append(ellipse)
        center_points.append(ellipse[0])
    
    if len(center_points) != total:
        return False, None, None
    
    # 如果提供了角点提示
    if corner_points_hint is not None and len(corner_points_hint) == 4:
        is_valid, sorted_ellipses, sorted_center_points = _sort_ellipses_and_center_points(
            pat_size, corner_points_hint, ellipses_filtered, center_points)
        
        if not is_valid:
            return False, None, None
        
        return True, sorted_center_points, sorted_ellipses
    
    # 找到四个角点
    corner_indices = _find_four_corners(center_points)
    if corner_indices is None:
        return False, None, None
    
    # 按逆时针排序
    corner_points_arr = np.array([center_points[i] for i in corner_indices], dtype=np.float32)
    hull = cv2.convexHull(corner_points_arr.reshape(-1, 1, 2), returnPoints=False)
    hull_indices = hull.flatten()
    corner_indices = [corner_indices[i] for i in hull_indices]
    
    # 找到左下角
    pt_image_bottom_left = np.array([0.0, img.shape[0]])
    idx_bottom_left = 0
    min_dist = 1e9
    
    for i in range(4):
        pt = center_points[corner_indices[i]]
        dist = np.hypot(pt[0] - pt_image_bottom_left[0], pt[1] - pt_image_bottom_left[1])
        if dist < min_dist:
            idx_bottom_left = i
            min_dist = dist
    
    # 按顺序排列角点
    corner_points = [center_points[corner_indices[(idx_bottom_left + i) % 4]] for i in range(4)]
    
    # 排序所有点
    is_valid, sorted_ellipses, sorted_center_points = _sort_ellipses_and_center_points(
        pat_size, corner_points, ellipses_filtered, center_points)
    
    if not is_valid:
        return False, None, None
    
    return True, sorted_center_points, sorted_ellipses
