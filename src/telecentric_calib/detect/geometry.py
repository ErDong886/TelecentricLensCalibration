"""标定板圆点检测。

移植自 Neucrede/find_calib_pattern，BSD-2-Clause License。
原作者 Neucrede <neucrede@sina.com>。
"""

import numpy as np
import cv2


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
