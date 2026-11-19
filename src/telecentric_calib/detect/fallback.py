"""圆网格与 HALCON 标定板的回退检测。"""
import numpy as np

from ..io import read_image
from .circles import FindCirclesGridPattern
from .halcon_board import FindHalconCalibBoard


def find_centers(image, pat_size):
    """先找普通圆网格，失败后再找 HALCON 标定板。"""
    success, centers, ellipses = FindCirclesGridPattern(image, pat_size)
    detection_method = None

    if not success:
        for thresh in range(10, 250, 10):
            success, centers, ellipses = FindCirclesGridPattern(image, pat_size, thresh)
            if success:
                detection_method = f"FindCirclesGridPattern (阈值={thresh})"
                break
    else:
        detection_method = "FindCirclesGridPattern (自动阈值)"

    if not success:
        success, centers, ellipses = FindHalconCalibBoard(image, pat_size)
        if not success:
            for thresh in range(10, 250, 10):
                success, centers, ellipses = FindHalconCalibBoard(image, pat_size, thresh)
                if success:
                    detection_method = f"FindHalconCalibBoard (阈值={thresh})"
                    break
        else:
            detection_method = "FindHalconCalibBoard (自动阈值)"

    return success, centers, ellipses, detection_method


def detect_pattern(image, pat_size, auto_threshold=True):
    """返回检测是否成功、中心点和椭圆。"""
    if not auto_threshold:
        success, centers, ellipses = FindCirclesGridPattern(image, pat_size)
        if not success:
            success, centers, ellipses = FindHalconCalibBoard(image, pat_size)
        return success, centers, ellipses
    success, centers, ellipses, _ = find_centers(image, pat_size)
    return success, centers, ellipses


def detect_pattern_from_path(image_path, pat_size, auto_threshold=True):
    """从图像路径检测标定板。"""
    image = read_image(image_path)
    return detect_pattern(image, pat_size, auto_threshold)


def detect_images(image_paths, board, log):
    """检测一组图像，返回世界坐标和像素坐标。"""
    pat_size = board.shape
    log("\n🔍 检测标定板...")
    log(f"   标定板尺寸: {pat_size[0]}x{pat_size[1]}")
    log(f"   找到 {len(image_paths)} 张图像")

    points_world = []
    points_pixel = []
    for idx, img_file in enumerate(image_paths):
        try:
            image = read_image(img_file)
        except Exception as exc:
            log(f"⚠️  无法读取图像: {img_file} - {exc}")
            continue

        success, centers, _, detection_method = find_centers(image, pat_size)
        if success:
            centers_array = np.array(centers, dtype=np.float32).reshape(-1, 2)
            points_pixel.append(centers_array)
            points_world.append(board.generate_world_points_grid())
            log(f"   ✓ 图像 {idx + 1}/{len(image_paths)}: 检测到 {len(centers)} 个标记点 [{detection_method}]")
        else:
            log(f"   ✗ 图像 {idx + 1}/{len(image_paths)}: 检测失败")

    if len(points_pixel) == 0:
        raise ValueError("所有图像的角点检测都失败了！")

    log(f"   📊 成功检测 {len(points_pixel)}/{len(image_paths)} 张图像")
    return points_world, points_pixel
