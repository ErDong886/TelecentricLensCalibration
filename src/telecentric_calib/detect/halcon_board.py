"""标定板圆点检测。

移植自 Neucrede/find_calib_pattern，BSD-2-Clause License。
原作者 Neucrede <neucrede@sina.com>。
"""

import numpy as np
import cv2

from .ellipse import fit_ellipse_subpixel
from .geometry import (
    _find_four_corners,
    _find_nearest_point,
    _sort_ellipses_and_center_points,
)


def _extract_contours_halcon_calib_board(img_gray, thresh, total, inverse_thresh=False):
    """提取Halcon标定板的轮廓"""
    # 二值化
    if thresh < 0:
        thresh_type = cv2.THRESH_BINARY_INV if inverse_thresh else cv2.THRESH_BINARY
        _, img_mono = cv2.threshold(img_gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        thresh_type = cv2.THRESH_BINARY_INV if inverse_thresh else cv2.THRESH_BINARY
        _, img_mono = cv2.threshold(img_gray, thresh, 255, thresh_type)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img_mono, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if hierarchy is None or len(contours) == 0:
        return None
    
    hierarchy = hierarchy[0]
    num_contours = len(contours)
    
    # 查找内外轮廓
    idx_inner = -1
    outer_contour = None
    inner_contour = None
    
    for i in range(num_contours):
        # 计算子轮廓数量
        n = 0
        j = hierarchy[i][2]
        while j >= 0:
            n += 1
            j = hierarchy[j][0]
        
        # 内轮廓应包含total个圆，且必须有有效的父轮廓
        if n < total or n > total + 5 or hierarchy[i][3] < 0:
            continue
        
        # 近似外轮廓
        parent_idx = hierarchy[i][3]
        eps = cv2.arcLength(contours[parent_idx], True) / 32.0
        outer_contour_approx = cv2.approxPolyDP(contours[parent_idx], eps, True)
        
        corner_indices = _find_four_corners(outer_contour_approx.reshape(-1, 2))
        if corner_indices is None:
            continue
        
        outer_contour = [outer_contour_approx[idx][0] for idx in corner_indices]
        
        # 近似内轮廓
        approx_points_inner = cv2.approxPolyDP(contours[i], 3, True)
        
        # Halcon标定板的内边框是带倒角的矩形，理想顶点数为5
        if len(approx_points_inner) < 5:
            continue
        
        idx_inner = i
        inner_contour = approx_points_inner.reshape(-1, 2).tolist()
        break
    
    if idx_inner < 0:
        return None
    
    # 计算blob区域
    blob_indices = []
    blob_areas = []
    
    for i in range(num_contours):
        if hierarchy[i][3] != idx_inner:
            continue
        
        contour = contours[i]
        moments = cv2.moments(contour)
        area = moments['m00']
        
        # 按面积过滤
        area_thresh = img_gray.shape[0] * img_gray.shape[1] / total
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
        
        if len(contour) >= 9:
            blob_indices.append(i)
            blob_areas.append(cv2.contourArea(contour))
    
    # 计算blob面积的均值和标准差
    num_blobs = len(blob_indices)
    if num_blobs == 0:
        return None
    
    mean_area = np.mean(blob_areas)
    stddev_area = np.std(blob_areas)
    
    # 使用3-sigma规则过滤
    area1 = mean_area - max(0.25 * mean_area, 3.0 * stddev_area)
    area2 = mean_area + max(0.25 * mean_area, 3.0 * stddev_area)
    
    blob_indices_filtered = []
    for i in range(num_blobs):
        if area1 <= blob_areas[i] <= area2:
            blob_indices_filtered.append(blob_indices[i])
    
    if len(blob_indices_filtered) != total:
        return None
    
    return {
        'contours': contours,
        'outer_contour': outer_contour,
        'inner_contour': inner_contour,
        'blob_indices_filtered': blob_indices_filtered
    }


def FindHalconCalibBoard(img, pat_size=(7, 7), thresh=-1, sub_pixel=True):
    """
    查找Halcon标定板
    
    参数:
        img: 输入图像
        pat_size: 标定板尺寸 (width, height)
        thresh: 二值化阈值，-1表示自动
        sub_pixel: 是否使用亚像素精度
    
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
    
    total = pat_size[0] * pat_size[1]
    
    # 提取Halcon标定板轮廓
    result = _extract_contours_halcon_calib_board(img_gray, thresh, total)
    if result is None:
        return False, None, None
    
    contours = result['contours']
    outer_contour = result['outer_contour']
    inner_contour = result['inner_contour']
    blob_indices_filtered = result['blob_indices_filtered']
    
    # 计算图像梯度
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    
    # 拟合椭圆并计算统计信息
    ellipses = []
    minor_lengths = []
    major_lengths = []
    
    for idx in blob_indices_filtered:
        contour = contours[idx]
        
        if sub_pixel:
            success, ellipse = fit_ellipse_subpixel(grad_x, grad_y, contour.reshape(-1, 2))
            if not success:
                return False, None, None
        else:
            ellipse = cv2.fitEllipseDirect(contour)
            bound_rect = cv2.boundingRect(contour)
            if bound_rect[2] * bound_rect[3] == 0:
                return False, None, None
        
        ellipses.append(ellipse)
        minor_length = min(ellipse[1][0], ellipse[1][1])
        major_length = max(ellipse[1][0], ellipse[1][1])
        minor_lengths.append(minor_length)
        major_lengths.append(major_length)
    
    mean_minor = np.mean(minor_lengths)
    mean_major = np.mean(major_lengths)
    stddev_minor = np.std(minor_lengths)
    stddev_major = np.std(major_lengths)
    
    # 有效长度范围
    low_min_len = mean_minor - max(0.25 * mean_minor, 3.0 * stddev_minor)
    upp_min_len = mean_minor + max(0.25 * mean_minor, 3.0 * stddev_minor)
    low_maj_len = mean_major - max(0.25 * mean_major, 3.0 * stddev_major)
    upp_maj_len = mean_major + max(0.25 * mean_major, 3.0 * stddev_major)
    
    # 过滤并获取中心点
    center_points = []
    ellipses_filtered = []
    
    for ellipse in ellipses:
        minor_length = min(ellipse[1][0], ellipse[1][1])
        major_length = max(ellipse[1][0], ellipse[1][1])
        
        if not (low_min_len <= minor_length <= upp_min_len):
            return False, None, None
        if not (low_maj_len <= major_length <= upp_maj_len):
            return False, None, None
        
        ellipses_filtered.append(ellipse)
        center_points.append(ellipse[0])
    
    # 将角点映射到矩形网格
    rectified_outer_points = np.array([
        [0, 0],
        [pat_size[0], 0],
        [pat_size[0], pat_size[1]],
        [0, pat_size[0]]
    ], dtype=np.float32)
    
    outer_contour_array = np.array(outer_contour, dtype=np.float32)
    H = cv2.findHomography(outer_contour_array, rectified_outer_points, 0)[0]
    
    inner_contour_array = np.array(inner_contour, dtype=np.float64).reshape(-1, 1, 2)
    rectified_inner_points = cv2.perspectiveTransform(inner_contour_array, H).reshape(-1, 2)
    
    # 找到距离倒角最近的外角
    idx0 = 0
    max_dist = 0
    
    for j in range(4):
        pt = rectified_outer_points[j]
        
        # 找最近点
        min_inner_dist = min(np.hypot(pt[0] - ipt[0], pt[1] - ipt[1]) 
                            for ipt in rectified_inner_points)
        
        if min_inner_dist > max_dist:
            idx0 = j
            max_dist = min_inner_dist
    
    pt_outer0 = tuple(outer_contour[idx0])
    idx_outer0 = idx0
    
    # 找到距离倒角最近的中心点作为原点
    idx0 = _find_nearest_point(center_points, pt_outer0)
    if idx0 < 0:
        return False, None, None
    
    pt_origin = center_points[idx0]
    
    # 找到X和Y轴上的外轮廓点
    pt_outer_a = outer_contour[(idx_outer0 + 1) % 4]
    pt_outer_b = outer_contour[(idx_outer0 + 3) % 4]
    
    vec_a = np.array([pt_outer_a[0] - pt_outer0[0], pt_outer_a[1] - pt_outer0[1]])
    vec_b = np.array([pt_outer_b[0] - pt_outer0[0], pt_outer_b[1] - pt_outer0[1]])
    cross_product = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
    
    if cross_product > 0:
        pt_outer1 = pt_outer_b
        pt_outer2 = pt_outer_a
    else:
        pt_outer1 = pt_outer_a
        pt_outer2 = pt_outer_b
    
    # 找到四个角点
    corner_indices = _find_four_corners(center_points)
    if corner_indices is None:
        return False, None, None
    
    # 排序角点
    sorted_corner_indices = [
        _find_nearest_point(center_points, pt_origin, indices=corner_indices),
        _find_nearest_point(center_points, pt_outer1, indices=corner_indices),
        _find_nearest_point(center_points, outer_contour[(idx_outer0 + 2) % 4], indices=corner_indices),
        _find_nearest_point(center_points, pt_outer2, indices=corner_indices)
    ]
    
    corner_points = [center_points[i] for i in sorted_corner_indices]
    
    # 排序所有点
    is_valid, sorted_ellipses, sorted_center_points = _sort_ellipses_and_center_points(
        pat_size, corner_points, ellipses_filtered, center_points)
    
    if not is_valid:
        return False, None, None
    
    return True, sorted_center_points, sorted_ellipses
