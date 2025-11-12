"""
相机标定API
提供高级封装接口
"""
import time
from ..src.calibrator import TelecentricCalibrator
from ..src.params import CalibParamHolder
from ..src.detector import PatternBoard


def run_calibration(
    img_dir: str, 
    pattern_info: PatternBoard, 
    magnification: float = 1.0, 
    visualization: bool = False
):
    """
    高层封装：给定图片目录与标定板信息，返回所有图片的世界点与像素点集合。

    参数:
        img_dir: 图片目录
        pattern_info: 标定板信息对象
        magnification: 远心镜头放大倍数（暂未使用）
        visualization: 是否可视化控制点检测（暂未使用）

    返回:
        (points_world, points_pixel)
            points_world: List[np.ndarray (N,3)]
            points_pixel: List[np.ndarray (N,2)]
    """
    # 创建参数容器
    param_holder = CalibParamHolder()
    
    # 从PatternBoard获取标定板参数
    board_params = {
        'dot_array_shape': pattern_info.shape,
        'dot_diameter': pattern_info.radius * 2 if pattern_info.radius else None,
        'dot_spacing': pattern_info.distance
    }
    param_holder.set_board(**board_params)
    
    # 创建标定器
    calibrator = TelecentricCalibrator(param_holder, img_dir)
    
    # 执行标定，获取世界坐标点和像素坐标点
    start_time = time.time()
    points_world, points_pixel = calibrator._step0_detect_corners()
    elapsed_time = time.time() - start_time
    
    if visualization:
        print(f"标定完成，耗时: {elapsed_time:.2f}秒")
    
    return points_world, points_pixel


def run_full_calibration(
    img_dir: str,
    pattern_info: PatternBoard,
    output_path: str = None
):
    """
    完整标定流程：包括角点检测、参数计算和优化

    参数:
        img_dir: 图片目录
        pattern_info: 标定板信息对象
        output_path: 输出JSON文件路径（可选）

    返回:
        final_params: 标定后的参数
    """
    # 创建参数容器
    param_holder = CalibParamHolder()
    
    # 从PatternBoard获取标定板参数
    board_params = {
        'dot_array_shape': pattern_info.shape,
        'dot_diameter': pattern_info.radius * 2 if pattern_info.radius else None,
        'dot_spacing': pattern_info.distance
    }
    param_holder.set_board(**board_params)
    
    # 创建标定器
    calibrator = TelecentricCalibrator(param_holder, img_dir)
    
    # 执行完整标定
    final_params = calibrator.single_camera_calibration()
    
    # 如果指定了输出路径，保存参数
    if output_path:
        calibrator.save_camera_params_to_json(output_path, image_index=0)
    
    return final_params

