#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本 - 不依赖命令行参数
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from telecentric_calib.detect import FindCirclesGridPattern, FindHalconCalibBoard


def main():
    # 图像路径
    image_path = r"D:\暂存\20251101\寻找halcon标定板\find_calib_pattern\test\2.bmp"
    
    print("=" * 60)
    print("标定板检测测试")
    print("=" * 60)
    print(f"\n加载图像: {image_path}")
    
    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误: 无法读取图像!")
        return 1
    
    print(f"图像尺寸: {frame.shape[1]} x {frame.shape[0]}")
    
    pat_size = (7, 7)
    print(f"\n检测 {pat_size[0]}x{pat_size[1]} 标定板...")
    print("-" * 60)
    
    # 方法1: FindCirclesGridPattern
    print("\n[测试1] FindCirclesGridPattern (自动阈值)")
    try:
        success, centers, ellipses = FindCirclesGridPattern(frame, pat_size)
        if success:
            print(f"✓ 检测成功! 找到 {len(centers)} 个标记点")
            print(f"  第一个点: ({centers[0][0]:.2f}, {centers[0][1]:.2f})")
            print(f"  最后一个点: ({centers[-1][0]:.2f}, {centers[-1][1]:.2f})")
        else:
            print("✗ 检测失败")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法2: FindCirclesGridPattern with threshold
    if not success:
        print("\n[测试2] FindCirclesGridPattern (手动阈值)")
        for thresh in [50, 100, 150, 200]:
            print(f"  尝试阈值 {thresh}...", end=" ")
            try:
                success, centers, ellipses = FindCirclesGridPattern(frame, pat_size, thresh)
                if success:
                    print(f"✓ 成功!")
                    break
                else:
                    print("✗ 失败")
            except Exception as e:
                print(f"✗ 错误: {e}")
    
    # 方法3: FindHalconCalibBoard
    if not success:
        print("\n[测试3] FindHalconCalibBoard (自动阈值)")
        try:
            success, centers, ellipses = FindHalconCalibBoard(frame, pat_size)
            if success:
                print(f"✓ 检测成功! 找到 {len(centers)} 个标记点")
                print(f"  第一个点: ({centers[0][0]:.2f}, {centers[0][1]:.2f})")
                print(f"  最后一个点: ({centers[-1][0]:.2f}, {centers[-1][1]:.2f})")
            else:
                print("✗ 检测失败")
        except Exception as e:
            print(f"✗ 发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 方法4: FindHalconCalibBoard with threshold
    if not success:
        print("\n[测试4] FindHalconCalibBoard (手动阈值)")
        for thresh in [50, 100, 150, 200]:
            print(f"  尝试阈值 {thresh}...", end=" ")
            try:
                success, centers, ellipses = FindHalconCalibBoard(frame, pat_size, thresh)
                if success:
                    print(f"✓ 成功!")
                    break
                else:
                    print("✗ 失败")
            except Exception as e:
                print(f"✗ 错误: {e}")
    
    print("\n" + "=" * 60)
    if success:
        print("最终结果: ✓ 检测成功!")
        print(f"找到 {len(centers)} 个标记点")
        
        # 绘制结果
        result_img = frame.copy()
        for i, pt in enumerate(centers):
            pt_int = tuple(map(int, pt))
            if i == 0:
                cv2.drawMarker(result_img, pt_int, (0, 255, 0), cv2.MARKER_STAR, 20, 2)
            else:
                cv2.circle(result_img, pt_int, 3, (0, 255, 0), -1)
        
        # 保存结果
        output_path = "detection_result.png"
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存到: {output_path}")
        
        # 显示结果
        try:
            cv2.imshow("Detection Result", result_img)
            print("\n按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("无法显示图像窗口（可能是无图形界面环境）")
    else:
        print("最终结果: ✗ 检测失败")
    
    print("=" * 60)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

