"""
独立测试脚本 - 可以直接运行

使用方法:
    python run_test.py <图像文件或文件夹路径>
    
示例:
    python run_test.py ../find_calib_pattern/test/2.bmp          # 单个文件
    python run_test.py ../find_calib_pattern/test/               # 批量处理文件夹
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import glob

# 导入本地模块
from fit_ellipse_subpixel import fit_ellipse_subpixel
from find_calib_pattern import FindCirclesGridPattern, FindHalconCalibBoard


WINDOW_NAME = "标定板检测结果"


def draw_results(frame, centers, pat_size):
    """在图像上绘制检测结果（不打印坐标）"""
    N = pat_size[0] * pat_size[1]
    
    for i in range(N):
        pt = tuple(map(int, centers[i]))
        
        if i == 0:
            # 原点用星形标记
            cv2.drawMarker(frame, pt, (0, 255, 0), cv2.MARKER_STAR, 16, 2)
        elif i == 1:
            # 第二个点用菱形标记
            cv2.drawMarker(frame, pt, (0, 255, 0), cv2.MARKER_DIAMOND, 16, 2)
        else:
            # 其他点用十字标记
            cv2.drawMarker(frame, pt, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        
        # 连线
        if i != N - 1:
            pt_next = tuple(map(int, centers[(i + 1) % N]))
            cv2.line(frame, pt, pt_next, (255, 0, 0), 1)


def show_results(frame, centers, pat_size):
    """在图像上显示检测结果并打印坐标"""
    draw_results(frame, centers, pat_size)
    
    # 打印坐标
    print("\n检测到的标记点坐标:")
    for r in range(pat_size[1]):
        print(f"行 {r + 1}:")
        for c in range(pat_size[0]):
            pt = centers[pat_size[0] * r + c]
            print(f"  ({pt[0]:.3f}, {pt[1]:.3f})", end="")
            if (c + 1) % 4 == 0:
                print()
        print()


def detect_single_image(image_path, pat_size=(7, 7), save_result=True, show_image=True, verbose=True):
    """
    检测单个图像
    
    参数:
        image_path: 图像路径
        pat_size: 标定板尺寸
        save_result: 是否保存结果
        show_image: 是否显示图像
        verbose: 是否打印详细信息
    
    返回:
        success: 是否成功
        centers: 中心点坐标
        output_path: 输出文件路径
    """
    if verbose:
        print(f"\n处理图像: {os.path.basename(image_path)}")
    
    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        if verbose:
            print(f"  ✗ 无法读取图像")
        return False, None, None
    
    if verbose:
        print(f"  图像尺寸: {frame.shape[1]} x {frame.shape[0]}")
    
    # 方法1: 尝试通用圆形网格检测
    success, centers, ellipses = FindCirclesGridPattern(frame, pat_size)
    
    if not success:
        # 尝试不同的阈值
        for thresh in range(10, 250, 10):
            success, centers, ellipses = FindCirclesGridPattern(frame, pat_size, thresh)
            if success:
                if verbose:
                    print(f"  ✓ FindCirclesGridPattern 成功 (阈值={thresh})")
                break
    elif verbose:
        print(f"  ✓ FindCirclesGridPattern 成功 (自动阈值)")
    
    # 如果第一种方法失败，尝试Halcon方法
    if not success:
        success, centers, ellipses = FindHalconCalibBoard(frame, pat_size)
        
        if not success:
            for thresh in range(10, 250, 10):
                success, centers, ellipses = FindHalconCalibBoard(frame, pat_size, thresh)
                if success:
                    if verbose:
                        print(f"  ✓ FindHalconCalibBoard 成功 (阈值={thresh})")
                    break
        elif verbose:
            print(f"  ✓ FindHalconCalibBoard 成功 (自动阈值)")
    
    # 绘制结果
    if success:
        frame = cv2.imread(image_path)  # 重新读取原始图像
        draw_results(frame, centers, pat_size)
    else:
        if verbose:
            print(f"  ✗ 检测失败")
        cv2.putText(frame, "No pattern detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 保存结果
    output_path = None
    if save_result:
        # 生成输出文件名: input.bmp -> input_result.png
        path_obj = Path(image_path)
        output_filename = f"{path_obj.stem}_result.png"
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        cv2.imwrite(output_path, frame)
        if verbose:
            print(f"  结果已保存: {output_filename}")
    
    # 显示图像
    if show_image:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 800, 600)
        cv2.imshow(WINDOW_NAME, frame)
        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return success, centers, output_path


def batch_process_directory(directory_path, pat_size=(7, 7)):
    """
    批量处理文件夹中的所有.bmp图像
    
    参数:
        directory_path: 文件夹路径
        pat_size: 标定板尺寸
    
    返回:
        success_count: 成功数量
        fail_count: 失败数量
    """
    # 查找所有.bmp文件
    bmp_pattern = os.path.join(directory_path, "*.bmp")
    image_files = glob.glob(bmp_pattern)
    
    if not image_files:
        print(f"错误: 在 {directory_path} 中未找到.bmp文件")
        return 0, 0
    
    print("=" * 70)
    print(f"批量处理模式")
    print(f"目标文件夹: {directory_path}")
    print(f"找到 {len(image_files)} 个.bmp文件")
    print(f"标定板尺寸: {pat_size[0]}x{pat_size[1]}")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    results = []
    
    # 处理每个图像
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        success, centers, output_path = detect_single_image(
            image_path, 
            pat_size=pat_size, 
            save_result=True, 
            show_image=False,  # 批量模式不显示图像
            verbose=True
        )
        
        if success:
            success_count += 1
            results.append((os.path.basename(image_path), "✓ 成功", output_path))
        else:
            fail_count += 1
            results.append((os.path.basename(image_path), "✗ 失败", output_path))
    
    # 打印统计结果
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"总计: {len(image_files)} 个文件")
    print(f"成功: {success_count} 个 ({success_count/len(image_files)*100:.1f}%)")
    print(f"失败: {fail_count} 个 ({fail_count/len(image_files)*100:.1f}%)")
    print("\n详细结果:")
    print("-" * 70)
    
    for filename, status, output_path in results:
        print(f"  {status}  {filename}")
    
    print("=" * 70)
    
    return success_count, fail_count


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python run_test.py <图像文件或文件夹路径>")
        print("\n示例:")
        print("  单个文件: python run_test.py ../find_calib_pattern/test/2.bmp")
        print("  批量处理: python run_test.py ../find_calib_pattern/test/")
        return 1
    
    input_path = sys.argv[1]
    pat_size = (7, 7)
    
    # 检查是文件还是文件夹
    if os.path.isfile(input_path):
        # 单个文件模式
        print("=" * 70)
        print("单文件处理模式")
        print("=" * 70)
        
        success, centers, output_path = detect_single_image(
            input_path,
            pat_size=pat_size,
            save_result=True,
            show_image=True,
            verbose=True
        )
        
        if success:
            show_results(cv2.imread(output_path), centers, pat_size)
        
        return 0 if success else 1
        
    elif os.path.isdir(input_path):
        # 批量处理模式
        success_count, fail_count = batch_process_directory(input_path, pat_size)
        return 0 if fail_count == 0 else 1
        
    else:
        print(f"错误: 路径不存在: {input_path}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

