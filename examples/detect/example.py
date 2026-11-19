"""
示例脚本 - 演示如何使用标定板检测库

这个脚本展示了两种使用方式：
1. 通用圆形网格检测
2. Halcon标定板检测
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from telecentric_calib.pattern import FindCirclesGridPattern, FindHalconCalibBoard


def example_circles_grid():
    """示例1: 通用圆形网格检测"""
    print("=" * 60)
    print("示例1: 通用圆形网格检测")
    print("=" * 60)
    
    # 读取图像
    img_path = '../find_calib_pattern/test/1.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 检测标定板 (7x7 网格)
    pat_size = (7, 7)
    print(f"\n检测 {pat_size[0]}x{pat_size[1]} 标定板...")
    
    success, centers, ellipses = FindCirclesGridPattern(
        img,
        pat_size=pat_size,
        sub_pixel=True
    )
    
    if success:
        print(f"✓ 检测成功! 找到 {len(centers)} 个标记点")
        
        # 打印前5个点的坐标
        print("\n前5个标记点坐标:")
        for i in range(min(5, len(centers))):
            print(f"  点 {i}: ({centers[i][0]:.2f}, {centers[i][1]:.2f})")
        
        # 可视化结果
        result_img = img.copy()
        for i, pt in enumerate(centers):
            pt_int = tuple(map(int, pt))
            
            if i == 0:
                # 原点 - 绿色星形
                cv2.drawMarker(result_img, pt_int, (0, 255, 0), 
                             cv2.MARKER_STAR, 20, 2)
                cv2.putText(result_img, "Origin", 
                          (pt_int[0] + 10, pt_int[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif i == 1:
                # 第二点 - 黄色菱形
                cv2.drawMarker(result_img, pt_int, (0, 255, 255), 
                             cv2.MARKER_DIAMOND, 16, 2)
            else:
                # 其他点 - 蓝色十字
                cv2.drawMarker(result_img, pt_int, (255, 0, 0), 
                             cv2.MARKER_CROSS, 12, 1)
            
            # 标注序号
            cv2.putText(result_img, str(i), 
                       (pt_int[0] + 5, pt_int[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 绘制椭圆
        if ellipses:
            for ellipse in ellipses:
                cv2.ellipse(result_img, ellipse, (0, 0, 255), 1)
        
        # 显示结果
        cv2.imshow('Circles Grid Detection Result', result_img)
        cv2.imwrite('result_circles_grid.png', result_img)
        print("\n结果已保存到 result_circles_grid.png")
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ 检测失败")


def example_halcon_board():
    """示例2: Halcon标定板检测"""
    print("\n" + "=" * 60)
    print("示例2: Halcon标定板检测")
    print("=" * 60)
    
    # 读取图像
    img_path = '../find_calib_pattern/test/1.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    # 检测Halcon标定板
    pat_size = (7, 7)
    print(f"\n检测 {pat_size[0]}x{pat_size[1]} Halcon标定板...")
    
    success, centers, ellipses = FindHalconCalibBoard(
        img,
        pat_size=pat_size,
        sub_pixel=True
    )
    
    if success:
        print(f"✓ 检测成功! 找到 {len(centers)} 个标记点")
        
        # 打印统计信息
        centers_array = np.array(centers)
        print(f"\n坐标范围:")
        print(f"  X: {centers_array[:, 0].min():.2f} ~ {centers_array[:, 0].max():.2f}")
        print(f"  Y: {centers_array[:, 1].min():.2f} ~ {centers_array[:, 1].max():.2f}")
        
        # 可视化结果
        result_img = img.copy()
        
        # 绘制网格连线
        for r in range(pat_size[1]):
            for c in range(pat_size[0]):
                idx = r * pat_size[0] + c
                pt = tuple(map(int, centers[idx]))
                
                # 横向连线
                if c < pat_size[0] - 1:
                    pt_right = tuple(map(int, centers[idx + 1]))
                    cv2.line(result_img, pt, pt_right, (255, 0, 0), 1)
                
                # 纵向连线
                if r < pat_size[1] - 1:
                    pt_down = tuple(map(int, centers[idx + pat_size[0]]))
                    cv2.line(result_img, pt, pt_down, (255, 0, 0), 1)
        
        # 绘制标记点
        for i, pt in enumerate(centers):
            pt_int = tuple(map(int, pt))
            
            if i == 0:
                # 原点 - 红色星形
                cv2.drawMarker(result_img, pt_int, (0, 0, 255), 
                             cv2.MARKER_STAR, 20, 3)
            else:
                # 其他点 - 绿色圆形
                cv2.circle(result_img, pt_int, 3, (0, 255, 0), -1)
        
        # 标注角点
        corner_idx = [0, pat_size[0]-1, len(centers)-1, len(centers)-pat_size[0]]
        labels = ["Origin", "X-axis", "Diagonal", "Y-axis"]
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0)]
        
        for idx, label, color in zip(corner_idx, labels, colors):
            pt = tuple(map(int, centers[idx]))
            cv2.putText(result_img, label, 
                       (pt[0] + 15, pt[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示结果
        cv2.imshow('Halcon Board Detection Result', result_img)
        cv2.imwrite('result_halcon_board.png', result_img)
        print("\n结果已保存到 result_halcon_board.png")
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ 检测失败")


def example_with_custom_threshold():
    """示例3: 使用自定义阈值"""
    print("\n" + "=" * 60)
    print("示例3: 使用自定义阈值")
    print("=" * 60)
    
    img_path = '../find_calib_pattern/test/1.png'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    pat_size = (7, 7)
    
    # 尝试不同的阈值
    print("\n尝试不同的二值化阈值...")
    thresholds = [50, 100, 150, 200]
    
    for thresh in thresholds:
        print(f"\n阈值 = {thresh}:")
        success, centers, _ = FindCirclesGridPattern(
            img,
            pat_size=pat_size,
            thresh=thresh
        )
        
        if success:
            print(f"  ✓ 检测成功! 找到 {len(centers)} 个点")
        else:
            print(f"  ✗ 检测失败")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "标定板检测库 - 使用示例" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        # 示例1: 通用圆形网格检测
        example_circles_grid()
        
        # 示例2: Halcon标定板检测
        example_halcon_board()
        
        # 示例3: 自定义阈值
        example_with_custom_threshold()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



