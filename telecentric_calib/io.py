"""图像读取、标定结果 JSON 和检测结果可视化。"""
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


def list_calibration_images(image_path: str) -> List[str]:
    """列出一张图或一个目录中的标定图像。目录顺序与 os.listdir 一致。"""
    if os.path.isfile(image_path):
        return [image_path]
    return [
        os.path.join(image_path, name)
        for name in os.listdir(image_path)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]


def build_camera_params_dict(alpha, beta, gamma, k1, k2, rotation, translation) -> dict:
    """组装单相机标定 JSON。CK 矩阵只保留前三行。"""
    rotation_matrix = np.asarray(rotation)
    translation_vector = np.asarray(translation).reshape(-1)
    intrinsic_matrix = np.array([
        [alpha, gamma, 0, 0],
        [0, beta, 0, 0],
        [0, 0, 0, 1],
    ])
    extrinsic_matrix = np.array([
        [rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], translation_vector[0]],
        [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], translation_vector[1]],
        [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0],
        [0, 0, 0, 1],
    ])
    ck_matrix = intrinsic_matrix @ extrinsic_matrix
    return {
        "CK_matrix": ck_matrix[:3, :].tolist(),
        "intrinsics": {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        },
        "distortion": {
            "k1": float(k1),
            "k2": float(k2),
        },
        "rotation_matrix": rotation_matrix.tolist(),
        "translation": [float(translation_vector[0]), float(translation_vector[1])],
    }


def build_ck_matrix_dict(alpha, beta, gamma, k1, k2, rvec, tvec, rotation) -> tuple:
    """组装已知内参姿态的 CK JSON，并返回 CK 矩阵。"""
    rotation_matrix = np.asarray(rotation)
    translation_vector = np.asarray(tvec).reshape(-1)
    intrinsic_matrix = np.array([
        [alpha, gamma, 0, 0],
        [0, beta, 0, 0],
        [0, 0, 0, 1],
    ])
    extrinsic_matrix = np.array([
        [rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], translation_vector[0]],
        [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], translation_vector[1]],
        [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0],
        [0, 0, 0, 1],
    ])
    ck_matrix = intrinsic_matrix @ extrinsic_matrix
    data = {
        "CK_matrix": ck_matrix.tolist(),
        "intrinsics": {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        },
        "distortion": {
            "k1": float(k1),
            "k2": float(k2),
        },
        "rotation_matrix": rotation_matrix.tolist(),
        "rotation_vector": np.asarray(rvec).flatten().tolist(),
        "translation": [float(translation_vector[0]), float(translation_vector[1])],
    }
    return ck_matrix, data

