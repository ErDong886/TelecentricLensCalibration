"""
双相机标定示例
包含误差评价
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
    from telecentric_calib.calibrator import TelecentricCalibrator, DualCameraCalibrator
    from telecentric_calib.config import get_config
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """双相机标定示例"""
    try:
        import numpy as np
        
        # 加载配置文件
        config = get_config()
        print("=" * 60)
        print("双相机标定示例")
        print("=" * 60)
        
        # ========== 1. 创建左右相机的参数容器 ==========
        print("\n" + "=" * 60)
        print("步骤1: 创建参数容器")
        print("=" * 60)
        
        # 从配置文件读取标定板参数
        board_config = config.get('calibration_board')
        
        # 左相机参数容器
        left_param_holder = CalibParamHolder()
        left_param_holder.set_board(
            dot_array_shape=tuple(board_config['dot_array_shape']),
            dot_diameter=board_config['dot_diameter'],
            dot_spacing=board_config['dot_spacing']
        )
        left_param_holder.set_intrinsic(
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
            k1=0.0,
            k2=0.0
        )
        left_param_holder.set_extrinsic(
            R=np.eye(3),
            T=np.zeros((3, 1))
        )
        print("✓ 左相机参数容器创建成功")
        
        # 右相机参数容器
        right_param_holder = CalibParamHolder()
        right_param_holder.set_board(
            dot_array_shape=tuple(board_config['dot_array_shape']),
            dot_diameter=board_config['dot_diameter'],
            dot_spacing=board_config['dot_spacing']
        )
        right_param_holder.set_intrinsic(
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
            k1=0.0,
            k2=0.0
        )
        right_param_holder.set_extrinsic(
            R=np.eye(3),
            T=np.zeros((3, 1))
        )
        print("✓ 右相机参数容器创建成功")
        
        # ========== 2. 从配置文件读取图像目录 ==========
        left_image_dir = config.get_path('left_camera_images')
        right_image_dir = config.get_path('right_camera_images')
        binocular_image_dir = config.get_path('binocular_images')
        
        print(f"\n左相机标定图像目录: {left_image_dir}")
        print(f"右相机标定图像目录: {right_image_dir}")
        print(f"双目标定图像目录: {binocular_image_dir}")
        
        # 检查目录是否存在
        for name, path in [("左相机", left_image_dir), ("右相机", right_image_dir), ("双目", binocular_image_dir)]:
            if not os.path.exists(path):
                print(f"✗ 错误: {name}图像目录不存在: {path}")
                print(f"   请检查配置文件中的路径设置或准备测试图像")
                return
            image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"✓ {name}目录: 找到 {len(image_files)} 个图像文件")
        
        # 获取verbose设置
        verbose = config.get('output.verbose', True)
        
        # ========== 3. 左相机标定 ==========
        print("\n" + "=" * 60)
        print("步骤2: 左相机标定")
        print("=" * 60)
        
        left_camera_calibrator = TelecentricCalibrator(left_param_holder, left_image_dir, verbose=verbose)
        print("✓ 左相机标定器创建成功")
        
        left_camera_calibrator.single_camera_calibration()
        print("✓ 左相机标定完成")
        
        # 保存左相机参数
        output_dir = config.get_path('output_dir', create=True)
        left_output_filename = config.get('output.left_camera_output', 'leftCKmatrix.json')
        left_output_path = os.path.join(output_dir, left_output_filename)
        left_camera_calibrator.save_camera_params_to_json(left_output_path, image_index=0)
        print(f"✓ 左相机参数已保存到: {left_output_path}")
        
        # ========== 4. 右相机标定 ==========
        print("\n" + "=" * 60)
        print("步骤3: 右相机标定")
        print("=" * 60)
        
        right_camera_calibrator = TelecentricCalibrator(right_param_holder, right_image_dir, verbose=verbose)
        print("✓ 右相机标定器创建成功")
        
        right_camera_calibrator.single_camera_calibration()
        print("✓ 右相机标定完成")
        
        # 保存右相机参数
        right_output_filename = config.get('output.right_camera_output', 'rightCKmatrix.json')
        right_output_path = os.path.join(output_dir, right_output_filename)
        right_camera_calibrator.save_camera_params_to_json(right_output_path, image_index=0)
        print(f"✓ 右相机参数已保存到: {right_output_path}")
        
        # ========== 5. 双目标定 ==========
        print("\n" + "=" * 60)
        print("步骤4: 双目标定和三维重建")
        print("=" * 60)
        
        # 从配置文件读取双目图像文件名
        left_image_name = config.get('binocular.left_image_name', '0_left.bmp')
        right_image_name = config.get('binocular.right_image_name', '0_right.bmp')
        
        dual_camera_calibrator = DualCameraCalibrator(
            left_camera_calibrator, 
            right_camera_calibrator, 
            binocular_image_dir,
            left_image_name=left_image_name,
            right_image_name=right_image_name,
            verbose=verbose
        )
        print("✓ 双目标定器创建成功")
        
        # 执行双目标定
        Mat_left, Mat_right, reconstructed_points, points_theory, results = \
            dual_camera_calibrator.dual_camera_calibration()
        print("✓ 双目标定完成")
        
        # ========== 6. 保存结果和误差评价 ==========
        print("\n" + "=" * 60)
        print("步骤5: 保存标定结果和误差评价")
        print("=" * 60)
        
        # 保存基础矩阵
        fundamental_matrix_filename = config.get('output.fundamental_matrix_output', 'fundamentalMatrix.json')
        fundamental_matrix_path = os.path.join(output_dir, fundamental_matrix_filename)
        dual_camera_calibrator.save_fundamental_matrix_json(
            results['points_pixel_left'], 
            results['points_pixel_right'], 
            fundamental_matrix_path
        )
        
        # 保存重建误差统计
        reconstruction_errors_filename = config.get('output.reconstruction_errors_output', 'reconstruction_errors.json')
        reconstruction_errors_path = os.path.join(output_dir, reconstruction_errors_filename)
        dual_camera_calibrator.save_reconstruction_errors_json(
            points_theory,
            results['distorted'],
            results['undistorted'],
            reconstruction_errors_path
        )
        
        # ========== 7. 误差评价总结 ==========
        print("\n" + "=" * 60)
        print("误差评价总结")
        print("=" * 60)
        
        errors_distorted = results['errors_distorted']
        errors_undistorted = results['errors_undistorted']
        
        print("\n【不带畸变校正的重建误差】")
        print(f"X方向 - 均值: {np.mean(errors_distorted[:, 0]):.6f} mm, "
              f"标准差: {np.std(errors_distorted[:, 0]):.6f} mm, "
              f"最大误差: {np.max(np.abs(errors_distorted[:, 0])):.6f} mm")
        print(f"Y方向 - 均值: {np.mean(errors_distorted[:, 1]):.6f} mm, "
              f"标准差: {np.std(errors_distorted[:, 1]):.6f} mm, "
              f"最大误差: {np.max(np.abs(errors_distorted[:, 1])):.6f} mm")
        print(f"Z方向 - 均值: {np.mean(errors_distorted[:, 2]):.6f} mm, "
              f"标准差: {np.std(errors_distorted[:, 2]):.6f} mm, "
              f"最大误差: {np.max(np.abs(errors_distorted[:, 2])):.6f} mm")
        rms_distorted = np.sqrt(np.mean(errors_distorted**2))
        print(f"整体RMS误差: {rms_distorted:.6f} mm")
        
        print("\n【带畸变校正的重建误差】")
        print(f"X方向 - 均值: {np.mean(errors_undistorted[:, 0]):.6f} mm, "
              f"标准差: {np.std(errors_undistorted[:, 0]):.6f} mm, "
              f"最大误差: {np.max(np.abs(errors_undistorted[:, 0])):.6f} mm")
        print(f"Y方向 - 均值: {np.mean(errors_undistorted[:, 1]):.6f} mm, "
              f"标准差: {np.std(errors_undistorted[:, 1]):.6f} mm, "
              f"最大误差: {np.max(np.abs(errors_undistorted[:, 1])):.6f} mm")
        print(f"Z方向 - 均值: {np.mean(errors_undistorted[:, 2]):.6f} mm, "
              f"标准差: {np.std(errors_undistorted[:, 2]):.6f} mm, "
              f"最大误差: {np.max(np.abs(errors_undistorted[:, 2])):.6f} mm")
        rms_undistorted = np.sqrt(np.mean(errors_undistorted**2))
        print(f"整体RMS误差: {rms_undistorted:.6f} mm")
        
        improvement = (rms_distorted - rms_undistorted) / rms_distorted * 100
        print(f"\n精度提升: {improvement:.2f}%")
        
        print("\n✅ 所有标定结果已保存！")
        print(f"  - {left_output_path}: 左相机内参和外参")
        print(f"  - {right_output_path}: 右相机内参和外参")
        print(f"  - {fundamental_matrix_path}: 基础矩阵")
        print(f"  - {reconstruction_errors_path}: 重建误差详细统计")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

