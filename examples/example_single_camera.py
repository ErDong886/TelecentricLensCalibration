"""
单相机标定示例
使用配置文件进行标定
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from telecentric_calib.params import CalibParamHolder
    from telecentric_calib.single import TelecentricCalibrator
    from telecentric_calib.config import get_config
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """单相机标定示例"""
    try:
        # 加载配置文件
        config = get_config()
        print("=" * 60)
        print("单相机标定示例")
        print("=" * 60)
        
        # 创建参数容器
        param_holder = CalibParamHolder()
        
        # 从配置文件读取标定板参数
        board_config = config.get('calibration_board')
        param_holder.set_board(
            dot_array_shape=tuple(board_config['dot_array_shape']),
            dot_diameter=board_config['dot_diameter'],
            dot_spacing=board_config['dot_spacing']
        )
        print("✓ 标定板参数已从配置文件加载")
        
        # 设置初始内参（可选，标定过程中会被优化）
        param_holder.set_intrinsic(
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
            k1=0.0,
            k2=0.0
        )
        
        # 设置初始外参（可选）
        param_holder.set_extrinsic(
            R=np.eye(3),
            T=np.zeros((3, 1))
        )
        
        # 从配置文件读取图像目录
        image_dir = config.get_path('left_camera_images')
        print(f"\n图像目录: {image_dir}")
        
        if not os.path.exists(image_dir):
            print(f"✗ 错误: 图像目录不存在: {image_dir}")
            print(f"   请检查配置文件中的路径设置或准备测试图像")
            return
        
        # 检查图像文件
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            print(f"✗ 错误: 在目录 {image_dir} 中未找到图像文件")
            return
        print(f"✓ 找到 {len(image_files)} 个图像文件")
        
        # 获取verbose设置
        verbose = config.get('output.verbose', True)
        
        # 创建标定器
        print("\n" + "=" * 60)
        print("开始单相机标定")
        print("=" * 60)
        calibrator = TelecentricCalibrator(param_holder, image_dir, verbose=verbose)
        print("✓ 标定器创建成功")
        
        # 执行标定
        print("\n开始执行标定...")
        calibrator.single_camera_calibration()
        print("✓ 标定完成")
        
        # 保存标定结果
        output_dir = config.get_path('output_dir', create=True)
        output_filename = config.get('output.left_camera_output', 'calibration_result.json')
        output_path = os.path.join(output_dir, output_filename)
        calibrator.save_camera_params_to_json(output_path, image_index=0)
        
        print("\n✅ 标定完成！")
        print(f"结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

