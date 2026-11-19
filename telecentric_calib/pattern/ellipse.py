"""
fit_ellipse_subpixel.py

亚像素精度椭圆拟合
基于Canny/Devernay算法实现

Reference:
Rafael Grompone von Gioi, Gregory Randall,
A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm,
Image Processing On Line, 7 (2017), pp. 347-372.
https://doi.org/10.5201/ipol.2017.216

Author: Converted from C++ version by Neucrede <neucrede@sina.com>
"""

import numpy as np
import cv2


def _get_clipped_integral_point(pt, clip_x, clip_y):
    """将浮点坐标转换为裁剪后的整数坐标"""
    x = int(pt[0])
    y = int(pt[1])
    
    if x < 0:
        x = 0
    elif x >= clip_x:
        x = clip_x - 1
    
    if y < 0:
        y = 0
    elif y >= clip_y:
        y = clip_y - 1
    
    return (x, y)


def _unchain_edge_point(chain, idx_from, idx_to):
    """解除边缘点链接"""
    if idx_from >= 0:
        chain[idx_from][0] = -1
    
    if idx_to >= 0:
        chain[idx_to][1] = -1


def _chain_edge_point(chain, idx_from, idx_to):
    """链接边缘点"""
    if chain[idx_from][0] >= 0:
        _unchain_edge_point(chain, idx_from, chain[idx_from][0])
    
    if chain[idx_to][1] >= 0:
        _unchain_edge_point(chain, chain[idx_to][1], idx_to)
    
    # 链接 FROM-->TO
    chain[idx_from][0] = idx_to
    chain[idx_to][1] = idx_from


def fit_ellipse_subpixel(grad_x, grad_y, points):
    """
    使用亚像素精度拟合椭圆
    
    参数:
        grad_x: 水平方向梯度图 (float32)
        grad_y: 垂直方向梯度图 (float32)
        points: 轮廓点列表 [(x, y), ...]
    
    返回:
        success: 是否成功
        ellipse: ((center_x, center_y), (width, height), angle)
    """
    neighbour_size = 5
    
    if len(points) < 6:
        return False, None
    
    # 转换点格式
    points = np.array(points, dtype=np.int32)
    
    # 计算周长和面积
    perim = cv2.arcLength(points, True)
    area = cv2.contourArea(points, False)
    circularity = 4 * np.pi * area / (perim * perim)
    if circularity < 0.5:
        return False, None
    
    # 初始椭圆拟合
    if len(points) < 6:
        return False, None
    
    ellipse0 = cv2.fitEllipseDirect(points.reshape(-1, 1, 2))
    bound_rect0 = cv2.boundingRect(points)
    if bound_rect0[2] < 8 or bound_rect0[3] < 8:
        return False, None
    
    # 扩展边界矩形
    x0 = max(0, bound_rect0[0] - neighbour_size)
    y0 = max(0, bound_rect0[1] - neighbour_size)
    x1 = min(grad_x.shape[1], bound_rect0[0] + bound_rect0[2] + neighbour_size)
    y1 = min(grad_x.shape[0], bound_rect0[1] + bound_rect0[3] + neighbour_size)
    
    if x1 <= x0 or y1 <= y0:
        return False, None
    
    # 创建掩码
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for pt in points:
        px = min(x1 - x0 - 1, max(0, pt[0] - x0))
        py = min(y1 - y0 - 1, max(0, pt[1] - y0))
        mask[py, px] = 255
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel)
    
    grad_x_roi = grad_x[y0:y1, x0:x1]
    grad_y_roi = grad_y[y0:y1, x0:x1]
    
    # 计算梯度幅值阈值
    max_mag = np.max(np.sqrt(grad_x_roi**2 + grad_y_roi**2))
    lower_mag_thresh = 0.5 * max_mag
    upper_mag_thresh = 0.75 * max_mag
    
    # 提取亚像素边缘点
    edge_points = []
    rows, cols = grad_x_roi.shape
    
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 0:
                continue
            
            mag_b = np.sqrt(grad_x_roi[r, c]**2 + grad_y_roi[r, c]**2)
            if mag_b < lower_mag_thresh:
                continue
            
            # 水平方向插值
            mag_a = np.sqrt(grad_x_roi[r, max(0, c-1)]**2 + grad_y_roi[r, max(0, c-1)]**2)
            mag_c = np.sqrt(grad_x_roi[r, min(cols-1, c+1)]**2 + grad_y_roi[r, min(cols-1, c+1)]**2)
            
            if (mag_a < mag_b and mag_b >= mag_c and 
                abs(grad_x_roi[r, c]) >= abs(grad_y_roi[r, c])):
                lambda_val = 0.5 * (mag_a - mag_c) / (mag_a - 2.0 * mag_b + mag_c + 1e-9)
                edge_points.append((c + lambda_val, r))
                continue
            
            # 垂直方向插值
            mag_a = np.sqrt(grad_x_roi[max(0, r-1), c]**2 + grad_y_roi[max(0, r-1), c]**2)
            mag_c = np.sqrt(grad_x_roi[min(rows-1, r+1), c]**2 + grad_y_roi[min(rows-1, r+1), c]**2)
            
            if (mag_a < mag_b and mag_b >= mag_c and 
                abs(grad_x_roi[r, c]) <= abs(grad_y_roi[r, c])):
                lambda_val = 0.5 * (mag_a - mag_c) / (mag_a - 2.0 * mag_b + mag_c + 1e-9)
                edge_points.append((c, r + lambda_val))
    
    if len(edge_points) < 6:
        return False, None
    
    # 创建边缘点索引映射
    index_map = np.full((rows, cols), -1, dtype=np.int32)
    for i, pt in enumerate(edge_points):
        int_pt = _get_clipped_integral_point(pt, cols, rows)
        if index_map[int_pt[1], int_pt[0]] == -1:
            index_map[int_pt[1], int_pt[0]] = i
    
    # 边缘点链接: [forward, prev, valid]
    chain = [[-1, -1, 0] for _ in range(len(edge_points))]
    
    for idx_p in range(len(edge_points)):
        pt_edge = edge_points[idx_p]
        pt = _get_clipped_integral_point(pt_edge, cols, rows)
        
        idx_best_forward = -1
        idx_best_backward = -1
        g_p = np.array([grad_x_roi[pt[1], pt[0]], grad_y_roi[pt[1], pt[0]]])
        min_dist_forward = 1e9
        min_dist_backward = 1e9
        
        # 在邻域内搜索
        for r in range(max(0, pt[1] - neighbour_size//2), 
                       min(rows-1, pt[1] + neighbour_size//2)):
            for c in range(max(0, pt[0] - neighbour_size//2),
                          min(cols-1, pt[0] + neighbour_size//2)):
                if r == pt[1] and c == pt[0]:
                    continue
                
                idx_n = index_map[r, c]
                if idx_n < 0:
                    continue
                
                pt_n = edge_points[idx_n]
                g_n = np.array([grad_x_roi[int(pt_n[1]), int(pt_n[0])], 
                               grad_y_roi[int(pt_n[1]), int(pt_n[0])]])
                g_n_ortho = np.array([-g_n[1], g_n[0]])  # 逆时针旋转90度
                vec_pn = np.array([pt_n[0] - pt[0], pt_n[1] - pt[1]])
                
                if np.dot(g_p, g_n) <= 0.0:
                    continue
                
                dist = np.hypot(vec_pn[0], vec_pn[1])
                
                if np.dot(vec_pn, g_n_ortho) > 0.0:
                    if dist < min_dist_forward:
                        idx_best_forward = idx_n
                        min_dist_forward = dist
                else:
                    if dist < min_dist_backward:
                        idx_best_backward = idx_n
                        min_dist_backward = dist
        
        # 链接前向点
        if idx_best_forward >= 0:
            if chain[idx_best_forward][1] < 0:
                _chain_edge_point(chain, idx_p, idx_best_forward)
            else:
                pt_bf = edge_points[idx_best_forward]
                dist1 = np.hypot(pt_bf[0] - pt_edge[0], pt_bf[1] - pt_edge[1])
                pt_bf_prev = edge_points[chain[idx_best_forward][1]]
                dist2 = np.hypot(pt_bf[0] - pt_bf_prev[0], pt_bf[1] - pt_bf_prev[1])
                if dist1 < dist2:
                    _chain_edge_point(chain, idx_p, idx_best_forward)
        
        # 链接后向点
        if idx_best_backward >= 0:
            if chain[idx_best_backward][0] < 0:
                _chain_edge_point(chain, idx_best_backward, idx_p)
            else:
                pt_bb = edge_points[idx_best_backward]
                dist1 = np.hypot(pt_bb[0] - pt_edge[0], pt_bb[1] - pt_edge[1])
                pt_bb_next = edge_points[chain[idx_best_backward][0]]
                dist2 = np.hypot(pt_bb[0] - pt_bb_next[0], pt_bb[1] - pt_bb_next[1])
                if dist1 < dist2:
                    _chain_edge_point(chain, idx_best_backward, idx_p)
    
    # 滞后阈值
    num_valid_edge_points = 0
    for idx in range(len(chain)):
        pt_edge = edge_points[idx]
        pt = _get_clipped_integral_point(pt_edge, cols, rows)
        mag = np.sqrt(grad_x_roi[pt[1], pt[0]]**2 + grad_y_roi[pt[1], pt[0]]**2)
        
        if chain[idx][2] == 0 and mag > upper_mag_thresh:
            chain[idx][2] = 1
            num_valid_edge_points += 1
            
            # 向前传播
            idx_next = chain[idx][0]
            while idx_next >= 0 and idx_next != idx and chain[idx_next][2] == 0:
                chain[idx_next][2] = 1
                num_valid_edge_points += 1
                idx_next = chain[idx_next][0]
            
            # 向后传播
            idx_prev = chain[idx][1]
            while idx_prev >= 0 and idx_prev != idx and chain[idx_prev][2] == 0:
                chain[idx_prev][2] = 1
                num_valid_edge_points += 1
                idx_prev = chain[idx_prev][1]
    
    # 收集所有有效的亚像素边缘点
    subpixel_points = []
    for idx in range(len(chain)):
        if chain[idx][2] == 1:
            pt_edge = edge_points[idx]
            subpixel_points.append((x0 + pt_edge[0], y0 + pt_edge[1]))
    
    if len(subpixel_points) < 6:
        return False, None
    
    # 拟合最终椭圆
    subpixel_points = np.array(subpixel_points, dtype=np.float32)
    ellipse = cv2.fitEllipseDirect(subpixel_points.reshape(-1, 1, 2))
    
    return True, ellipse



