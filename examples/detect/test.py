"""
test.py

标定板检测测试程序

支持两种模式:
1. 实时视频检测 (不带参数)
2. 静态图像检测 (带图像路径参数)

使用方法:
    python test.py                    # 实时视频模式
    python test.py image.png          # 静态图像模式
"""

import sys
import cv2
import numpy as np

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from telecentric_calib.detect import FindCirclesGridPattern, FindHalconCalibBoard


WINDOW_NAME = "FindHalconCalibBoard()"


def show_results(frame, centers, pat_size):
    """
    在图像上显示检测结果
    
    参数:
        frame: 输入图像
        centers: 中心点列表
        pat_size: 标定板尺寸
    """
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
    
    # 打印坐标
    for r in range(pat_size[1]):
        print(f"Row {r + 1}")
        for c in range(pat_size[0]):
            pt = centers[pat_size[0] * r + c]
            print(f"  ({pt[0]:.3f}, {pt[1]:.3f})", end="")
            if (c + 1) % 4 == 0:
                print()
        print()


def live_video_test():
    """实时视频测试"""
    print("启动实时视频检测...")
    print("按 'q' 键退出")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return 1
    
    pat_size = (7, 7)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("错误: 无法读取帧")
            continue
        
        frame_count += 1
        print(f"\n帧 #{frame_count}")
        
        # 先尝试自动阈值
        # success, centers, ellipses = FindCirclesGridPattern(frame, pat_size)
        success, centers, ellipses = FindHalconCalibBoard(frame, pat_size)
        
        if success:
            show_results(frame, centers, pat_size)
        else:
            # 尝试不同的阈值
            for thresh in range(10, 250, 10):
                # success, centers, ellipses = FindCirclesGridPattern(frame, pat_size, thresh)
                success, centers, ellipses = FindHalconCalibBoard(frame, pat_size, thresh)
                
                if success:
                    show_results(frame, centers, pat_size)
                    break
        
        if not success:
            cv2.putText(frame, "No pattern detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return 0


def still_image_test(image_path):
    """静态图像测试"""
    print(f"加载图像: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误: 无法读取图像 {image_path}")
        return 1
    
    pat_size = (7, 7)
    
    # 先尝试自动阈值
    success, centers, ellipses = FindCirclesGridPattern(frame, pat_size)
    # success, centers, ellipses = FindHalconCalibBoard(frame, pat_size)
    
    if success:
        print("检测成功!")
        show_results(frame, centers, pat_size)
    else:
        print("使用自动阈值检测失败，尝试不同阈值...")
        # 尝试不同的阈值
        for thresh in range(10, 250, 10):
            # success, centers, ellipses = FindCirclesGridPattern(frame, pat_size, thresh)
            success, centers, ellipses = FindHalconCalibBoard(frame, pat_size, thresh)
            
            if success:
                print(f"检测成功! (阈值={thresh})")
                show_results(frame, centers, pat_size)
                break
    
    if not success:
        print("检测失败")
        cv2.putText(frame, "No pattern detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow(WINDOW_NAME, frame)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0


def main():
    """主函数"""
    # 创建窗口
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    
    if len(sys.argv) > 1:
        # 静态图像模式
        return still_image_test(sys.argv[1])
    else:
        # 实时视频模式
        return live_video_test()


if __name__ == "__main__":
    sys.exit(main())



