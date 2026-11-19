"""
find_calib_pattern.py

用于检测和识别图像中的标定板圆形标记点

Author: Converted from C++ version by Neucrede <neucrede@sina.com>
License: BSD 2-Clause License
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

# 支持多种导入方式
try:
    from .fit_ellipse_subpixel import fit_ellipse_subpixel
except ImportError:
    from fit_ellipse_subpixel import fit_ellipse_subpixel


def _point_line_distance(pt, pt_line_a, pt_line_b):
    """计算点到直线的距离"""
    x_p = pt[0] - pt_line_a[0]
    y_p = pt[1] - pt_line_a[1]
    x_v = pt_line_b[0] - pt_line_a[0]
    y_v = pt_line_b[1] - pt_line_a[1]
    vec_len = np.hypot(x_v, y_v) + 1e-9
    cross_prod = x_p * y_v - x_v * y_p
    return abs(cross_prod) / vec_len


def _find_nearest_point(points, pt0, pt_line_a=None, pt_line_b=None, indices=None):
    """找到最近的点"""
    if indices is not None:
        search_points = [points[i] for i in indices]
    else:
        search_points = points
    
    idx = -1
    min_dist = 1e9
    
    for i, pt in enumerate(search_points):
        dist = np.hypot(pt[0] - pt0[0], pt[1] - pt0[1])
        if pt_line_a is not None and pt_line_b is not None:
            dist += _point_line_distance(pt, pt_line_a, pt_line_b)
        
        if dist < min_dist:
            if indices is not None:
                idx = indices[i]
            else:
                idx = i
            min_dist = dist
    
    return idx


def _find_four_corners(points):
    """找到四个角点"""
    points = np.array(points, dtype=np.float32)
    
    # 计算凸包
    hull_indices = cv2.convexHull(points.reshape(-1, 1, 2), returnPoints=False)
    hull_indices = hull_indices.flatten()
    
    if len(hull_indices) < 4:
        return None
    
    # 计算相邻三个顶点形成的角的余弦值
    angle_cosines = []
    K = len(hull_indices)
    
    for i in range(K):
        idx_curr = hull_indices[i]
        idx_next = hull_indices[(i + 1) % K]
        idx_prev = hull_indices[(i - 1 + K) % K]
        
        vec1 = points[idx_next] - points[idx_curr]
        vec2 = points[idx_prev] - points[idx_curr]
        
        cos_angle = abs(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))
        angle_cosines.append((cos_angle, hull_indices[i]))
    
    # 按角度余弦值排序，前4个最锐利的角为角点
    angle_cosines.sort(key=lambda x: x[0])
    corner_indices = sorted([x[1] for x in angle_cosines[:4]])
    
    return corner_indices


def _hierarchical_clustering(points, pattern_size):
    """层次聚类算法 - 从OpenCV移植"""
    n = len(points)
    pn = pattern_size[0] * pattern_size[1]
    
    if pn >= n:
        if pn == n:
            return list(range(n))
        else:
            return None
    
    # 计算距离矩阵
    dists = np.zeros((n, n), dtype=np.float64)
    dists_mask = np.zeros((n, n), dtype=np.uint8)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            dists[i, j] = dist
            dists[j, i] = dist
            dists_mask[i, j] = 255
            dists_mask[j, i] = 255
    
    # 初始化聚类
    clusters = [[i] for i in range(n)]
    
    pattern_cluster_idx = 0
    while len(clusters[pattern_cluster_idx]) < pn:
        # 找到最小距离
        masked_dists = np.where(dists_mask > 0, dists, np.inf)
        min_idx = np.unravel_index(np.argmin(masked_dists), masked_dists.shape)
        min_row = min(min_idx[0], min_idx[1])
        max_row = max(min_idx[0], min_idx[1])
        
        # 更新距离矩阵
        dists_mask[max_row, :] = 0
        dists_mask[:, max_row] = 0
        
        dists[min_row, :] = np.minimum(dists[min_idx[0], :], dists[min_idx[1], :])
        dists[:, min_row] = dists[min_row, :]
        
        # 合并聚类
        clusters[min_row].extend(clusters[max_row])
        clusters[max_row] = []
        pattern_cluster_idx = min_row
    
    if len(clusters[pattern_cluster_idx]) < pn:
        return None
    
    return clusters[pattern_cluster_idx]


def _sort_ellipses_and_center_points(pat_size, corner_points, ellipses, center_points):
    """排序椭圆和中心点"""
    # 校正中心点
    src_points = np.array(corner_points[:4], dtype=np.float64)
    
    len1 = np.linalg.norm(src_points[1] - src_points[0])
    len2 = np.linalg.norm(src_points[2] - src_points[0])
    stride = (len1 + len2) / (2 * (pat_size[0] + pat_size[1] - 2)) + 1.0
    
    dest_points = np.array([
        [0, 0],
        [(pat_size[0] - 1) * stride, 0],
        [(pat_size[0] - 1) * stride, (pat_size[1] - 1) * stride],
        [0, (pat_size[1] - 1) * stride]
    ], dtype=np.float64)
    
    H = cv2.findHomography(src_points, dest_points, 0)[0]
    
    center_points_array = np.array(center_points, dtype=np.float64).reshape(-1, 1, 2)
    rectified_points = cv2.perspectiveTransform(center_points_array, H).reshape(-1, 2)
    
    # 按字典序排序
    sorted_center_points = []
    sorted_ellipses = []
    indices = []
    
    for r in range(pat_size[1]):
        for c in range(pat_size[0]):
            pt_ideal = np.array([c * stride, r * stride])
            pt_a = np.array([0.0, r * stride])
            pt_b = np.array([(pat_size[0] - 1) * stride, r * stride])
            
            # 找到最近的点
            idx = -1
            min_dist = 1e9
            for i, pt in enumerate(rectified_points):
                dist = np.hypot(pt[0] - pt_ideal[0], pt[1] - pt_ideal[1])
                dist += _point_line_distance(pt, pt_a, pt_b)
                if dist < min_dist:
                    idx = i
                    min_dist = dist
            
            sorted_center_points.append(center_points[idx])
            sorted_ellipses.append(ellipses[idx])
            indices.append(idx)
    
    # 验证排序
    N = pat_size[0]
    pt0 = rectified_points[indices[0]]
    pt1 = rectified_points[indices[1]]
    ptN = rectified_points[indices[N - 1]]
    
    dist01 = np.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1])
    dist0N = np.hypot(pt0[0] - ptN[0], pt0[1] - ptN[1])
    
    is_valid = (dist0N > (N - 2) * dist01) and (dist0N < 1.2 * (N - 1) * dist01)
    
    return is_valid, sorted_ellipses, sorted_center_points


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



