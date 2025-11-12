"""
工具函数模块
包含图像IO、JSON处理、可视化等功能
"""
import cv2
import numpy as np
import json
import os
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def read_image(image_path: str) -> np.ndarray:
    """
    读取图像（处理中文路径）
    
    参数:
        image_path: 图像文件路径
    
    返回:
        图像数组 (BGR格式)
    """
    # 使用numpy读取，避免中文路径问题
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    return img


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    保存图像（处理中文路径）
    
    参数:
        image: 图像数组
        output_path: 输出文件路径
    
    返回:
        是否保存成功
    """
    # 使用imencode避免中文路径问题
    success, encoded_img = cv2.imencode('.bmp', image)
    if success:
        encoded_img.tofile(output_path)
    return success


def get_image_files(image_dir: str) -> List[str]:
    """
    获取目录下所有图像文件
    
    参数:
        image_dir: 图像目录路径
    
    返回:
        图像文件路径列表
    """
    if os.path.isfile(image_dir):
        return [image_dir]
    
    image_files = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_files.append(os.path.join(image_dir, f))
    
    return sorted(image_files)


def save_json(data: dict, output_path: str, indent: int = 4) -> None:
    """
    保存数据到JSON文件
    
    参数:
        data: 要保存的数据字典
        output_path: 输出文件路径
        indent: JSON缩进
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(json_path: str) -> dict:
    """
    从JSON文件加载数据
    
    参数:
        json_path: JSON文件路径
    
    返回:
        数据字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_detection(image: np.ndarray, centers: List[Tuple[float, float]], 
                       output_path: Optional[str] = None) -> np.ndarray:
    """
    可视化检测结果
    
    参数:
        image: 输入图像
        centers: 检测到的中心点列表
        output_path: 可选，保存路径
    
    返回:
        标注后的图像
    """
    img_vis = image.copy()
    
    for center in centers:
        cv2.circle(img_vis, (int(center[0]), int(center[1])), 5, (0, 255, 0), 2)
    
    if output_path:
        save_image(img_vis, output_path)
    
    return img_vis

