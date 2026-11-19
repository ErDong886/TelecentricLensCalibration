from .params import CalibParamHolder
from .pattern import FindCirclesGridPattern, FindHalconCalibBoard
from .utils import read_image
import cv2
import numpy as np
import json
import os

class TelecentricCalibrator:
    # 初始化标定器，传入参数容器和图像目录，以及是否可视化
    def __init__(self, param_holder:CalibParamHolder, image_dir:str, verbose:bool = True):
        """
        初始化标定器，参数:
            param_holder: 参数容器
            image_dir: 图像目录
            verbose: 是否显示详细输出（默认True）
        """
        self.param_holder = param_holder
        self.image_dir = image_dir
        self.verbose = verbose
        # 读取图像尺寸
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            raise ValueError(f"在目录 {image_dir} 中未找到图像文件")
        first_image_path = os.path.join(image_dir, image_files[0])
        first_image = read_image(first_image_path)
        self.image_size = (first_image.shape[1], first_image.shape[0])  # (width, height)
    
    def _print(self, *args, **kwargs):
        """条件输出方法，仅在verbose=True时输出"""
        if self.verbose:
            print(*args, **kwargs)
        
    # 单相机标定
    def single_camera_calibration(self):
        points_world, points_pixel = self._step0_detect_corners()
        H_matrices, A_matrix, Ks_list = self._step1_solve_H_A_Ks(points_world, points_pixel)
        magnitudes_list = self._step2_solve_r31_r32_magnitudes(Ks_list)
        signs_list = self._step3_determine_r31_r32_signs(points_world, points_pixel)
        optimal_R_list = self._step4_construct_rotation_matrices(Ks_list, magnitudes_list, signs_list)
        final_params = self._step5_global_optimization(points_world, points_pixel, H_matrices, A_matrix, Ks_list, optimal_R_list)
        # 保存参数到成员变量
        self.final_params = final_params
        return final_params
    
    def save_camera_params_to_json(self, output_path:str, image_index:int=0):
        """
        保存相机内参和外参到JSON文件（模仿leftCKmatrix_01.json格式）
        
        参数:
            output_path: 输出JSON文件路径
            image_index: 使用第几张图像的外参（默认第一张）
        """
        if self.final_params is None:
            raise ValueError("final_params未确定，请先执行single_camera_calibration")
        
        # 提取优化后的参数
        alpha, beta, gamma, k1, k2, R_list, tvecs = self.final_params
        
        # 选择指定索引的外参
        if image_index >= len(R_list):
            self._print(f"⚠️  警告：image_index={image_index}超出范围，使用第一张图像的外参")
            image_index = 0
        
        R = R_list[image_index]
        t = tvecs[image_index]
        
        # 构建CK矩阵（内参矩阵 × 外参矩阵的前3列）
        intrinsic_matrix = np.array([
            [alpha, gamma, 0, 0],
            [0, beta, 0, 0],
            [0, 0, 0, 1]
        ])
        
        extrinsic_matrix = np.array([
            [R[0, 0], R[0, 1], R[0, 2], t[0]],
            [R[1, 0], R[1, 1], R[1, 2], t[1]],
            [R[2, 0], R[2, 1], R[2, 2], 0],
            [0, 0, 0, 1]
        ])
        
        CK_matrix = intrinsic_matrix @ extrinsic_matrix
        
        # 构建输出字典
        output_dict = {
            "CK_matrix": CK_matrix[:3, :].tolist(),  # 只保存前3行
            "intrinsics": {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma)
            },
            "distortion": {
                "k1": float(k1),
                "k2": float(k2)
            },
            "rotation_matrix": R.tolist(),
            "translation": [float(t[0]), float(t[1])]
        }
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)
        
        self._print(f"✅ 相机参数已保存到: {output_path}")
        return output_dict
    
    # 使用已经确定的单相机参数（内参），标定单张照片的旋转和平移向量
    def known_intrinsic_calibration_single_image(self, image_path:str):
        # 检验final_params是否已经确定
        if self.final_params is None:
            raise ValueError("final_params未确定，请先确定单相机参数，使用single_camera_calibration方法")
        image = read_image(image_path)
        
        # 使用 pattern 包检测标定板圆点
        # 检测逻辑完全符合run_test.py的实现
        # FindHalconCalibBoard 和 FindCirclesGridPattern 已在文件开头导入
        board_params = self.param_holder.get_board_dict()
        pat_size = board_params['dot_array_shape']
        
        # ========== 完全符合run_test.py的检测逻辑 ==========
        # 方法1: 尝试通用圆形网格检测
        success, centers, ellipses = FindCirclesGridPattern(image, pat_size)
        detection_method = None
        
        if not success:
            # 尝试不同的阈值
            for thresh in range(10, 250, 10):
                success, centers, ellipses = FindCirclesGridPattern(image, pat_size, thresh)
                if success:
                    detection_method = f"FindCirclesGridPattern (阈值={thresh})"
                    break
        else:
            detection_method = "FindCirclesGridPattern (自动阈值)"
        
        # 如果第一种方法失败，尝试Halcon方法
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
        
        if not success:
            raise ValueError(f"角点检测失败: {image_path}")
        
        # print(f"   ✓ 角点检测成功 [{detection_method}]")
        
        # 转换为numpy数组格式 (N, 2)
        points_pixel_single = np.array(centers, dtype=np.float32).reshape(-1, 2)
        
        # 获取世界坐标点
        points_world_single = self.pattern_board.generate_world_points_grid()
        # 从final_params中获取内参
        alpha = self.final_params[0]
        beta = self.final_params[1]
        gamma = self.final_params[2]
        k1 = self.final_params[3]
        k2 = self.final_params[4]
        
        # print(f"\n📷 使用已知内参处理单张图像: {image_path}")
        # print(f"   内参: alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")
        # print(f"   畸变: k1={k1:.6e}, k2={k2:.6e}")
        
        # ========== Step 1: 计算单应性矩阵H ==========
        # 只取世界坐标的前两个分量（X, Y），因为Z=0
        src_pts = np.array(points_world_single)[:, :2].astype(np.float32)
        dst_pts = points_pixel_single.reshape(-1, 2).astype(np.float32)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)  # DLT算法
        if H is None:
            raise ValueError(f"单应性矩阵H计算失败: {image_path}")
        
        # 归一化H矩阵
        if H[2, 2] != 0:
            H = H / H[2, 2]
        
        # print(f"   单应性矩阵H:\n{H}")
        
        # ========== Step 2: 利用已知内参A计算截断外参Ks ==========
        # 构建内参矩阵A
        A_matrix = np.array([
            [alpha, gamma, 0],
            [0, beta, 0],
            [0, 0, 1]
        ])
        
        # Ks = A^{-1} * H
        A_inv = np.linalg.inv(A_matrix)
        Ks = A_inv @ H
        
        # print(f"   截断外参矩阵Ks:\n{Ks}")
        
        # ========== Step 3: 求解r31和r32的绝对值 ==========
        # 从Ks矩阵中提取已知的外参元素
        r11, r21, _ = Ks[:, 0]
        r12, r22, _ = Ks[:, 1]
        
        # 使用正交性和模长相等约束求解r31和r32
        from scipy.optimize import fsolve
        
        def equations(p):
            # 正交性约束: r11*r12 + r21*r22 + r31*r32 = 0
            eq1 = r11*r12 + r21*r22 + p[0]*p[1]
            # 模长相等约束: r11^2+r21^2+r31^2 = r12^2+r22^2+r32^2
            eq2 = (r11**2 + r21**2 + p[0]**2) - (r12**2 + r22**2 + p[1]**2)
            return (eq1, eq2)
        
        r31_solution, r32_solution = fsolve(equations, (0.01, 0.01))
        abs_r31 = np.abs(r31_solution)
        abs_r32 = np.abs(r32_solution)
        
        # ========== Step 4: 确定r31和r32的符号 ==========
        # 使用针孔模型作为参考
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [points_world_single],
            [points_pixel_single],
            self.image_size,
            None,
            None
        )
        
        if not ret:
            self._print("   ⚠️  针孔模型标定失败，使用默认符号")
            sign_r31, sign_r32 = 1.0, 1.0
        else:
            # 将旋转向量转换为旋转矩阵
            R_pinhole, _ = cv2.Rodrigues(rvecs[0])
            sign_r31 = np.sign(R_pinhole[2, 0]) if R_pinhole[2, 0] != 0 else 1.0
            sign_r32 = np.sign(R_pinhole[2, 1]) if R_pinhole[2, 1] != 0 else 1.0
        
        correct_r31 = abs_r31 * sign_r31
        correct_r32 = abs_r32 * sign_r32
        
        # ========== Step 5: 构造完整的旋转矩阵R ==========
        rx = np.array([r11, r21, correct_r31])
        ry = np.array([r12, r22, correct_r32])
        
        # 通过叉乘得到rz
        rz = np.cross(rx, ry)
        
        # 构造旋转矩阵R
        R_approx = np.column_stack([rx, ry, rz])
        
        # 使用SVD方法对R进行强制正交化
        U, _, Vt = np.linalg.svd(R_approx)
        R_ortho = U @ Vt
        
        # ========== Step 6: 提取平移向量 ==========
        # 从Ks的第三列提取前两个分量作为平移向量
        tx = Ks[0, 2]
        ty = Ks[1, 2]
        t_vec = np.array([tx, ty])
        
        # 将旋转矩阵转为旋转向量
        rvec, _ = cv2.Rodrigues(R_ortho)
        
        self._print("   ✅ 外参求解成功")
        self._print(f"   旋转矩阵R:\n{R_ortho}")
        self._print(f"   旋转向量rvec: {rvec.T}")
        self._print(f"   平移向量tvec: {t_vec}")
        self._print(f"   det(R): {np.linalg.det(R_ortho):.6f}")
        
        return rvec, t_vec, R_ortho
    
    def save_ck_matrix_json(self, rvec, tvec, R_matrix, output_path="CKmatrix.json"):
        """
        保存CK矩阵和内外参到JSON文件
        
        参数:
            rvec: 旋转向量
            tvec: 平移向量 (2D)
            R_matrix: 旋转矩阵
            output_path: 输出文件路径
        """
        # 获取内参
        alpha = self.param_holder.alpha
        beta = self.param_holder.beta
        gamma = self.param_holder.gamma
        k1 = self.param_holder.k1
        k2 = self.param_holder.k2
        
        # 构建内参矩阵 (3x4)
        intrinsic_matrix = np.array([
            [alpha, gamma, 0, 0],
            [0, beta, 0, 0],
            [0, 0, 0, 1]
        ])
        
        # 构建外参矩阵 (4x4)
        extrinsic_matrix = np.array([
            [R_matrix[0, 0], R_matrix[0, 1], R_matrix[0, 2], tvec[0]],
            [R_matrix[1, 0], R_matrix[1, 1], R_matrix[1, 2], tvec[1]],
            [R_matrix[2, 0], R_matrix[2, 1], R_matrix[2, 2], 0],
            [0, 0, 0, 1]
        ])
        
        # 计算CK矩阵 (3x4)
        ck_matrix = intrinsic_matrix @ extrinsic_matrix
        
        # 准备JSON数据
        data = {
            "CK_matrix": ck_matrix.tolist(),
            "intrinsics": {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma)
            },
            "distortion": {
                "k1": float(k1),
                "k2": float(k2)
            },
            "rotation_matrix": R_matrix.tolist(),
            "rotation_vector": rvec.flatten().tolist(),
            "translation": [float(tvec[0]), float(tvec[1])]
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        self._print(f"   ✅ CK矩阵已保存到: {output_path}")
        
        return ck_matrix
        
        
    #### 具体标定流程 ####
    def _step0_detect_corners(self, image_path = None):
        if image_path is None:
            image_path = self.image_dir
        """Step 0: 从image_dir读取图像，检测角点并且返回世界坐标点和像素坐标点"""
        def _create_pattern_board(self):
            """从参数容器创建标定板对象"""
            
            from .detector import PatternBoard
            
            board_params = self.param_holder.get_board_dict()
            
            # 验证参数
            if board_params['dot_array_shape'] is None:
                raise ValueError("标定板参数未设置！请先设置 dot_array_shape")
            if board_params['dot_diameter'] is None:
                raise ValueError("标定板参数未设置！请先设置 dot_diameter")
            if board_params['dot_spacing'] is None:
                raise ValueError("标定板参数未设置！请先设置 dot_spacing")
            
            return PatternBoard(
                pattern_type=1,
                shape=board_params['dot_array_shape'],
                distance=board_params['dot_spacing'],
                pixel_size=(3.45, 3.45),  # 固定像素尺寸
                radius=board_params['dot_diameter'] / 2
            )
            
        # 创建标定板对象
        self.pattern_board = _create_pattern_board(self)
        
        # 使用 pattern 包检测标定板圆点
        # 检测逻辑完全符合run_test.py的实现
        # FindHalconCalibBoard 和 FindCirclesGridPattern 已在文件开头导入
        
        # 获取标定板参数
        board_params = self.param_holder.get_board_dict()
        pat_size = board_params['dot_array_shape']
        
        # 获取图像目录下的所有图像文件
        if os.path.isfile(image_path):
            # 如果传入的是单个文件
            image_files = [image_path]
            base_dir = os.path.dirname(image_path)
        else:
            # 如果传入的是目录
            base_dir = image_path
            image_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"在目录 {image_path} 中未找到图像文件")
        
        points_world = []
        points_pixel = []
        
        self._print("\n🔍 检测标定板...")
        self._print(f"   标定板尺寸: {pat_size[0]}x{pat_size[1]}")
        self._print(f"   找到 {len(image_files)} 张图像")
        
        for idx, img_file in enumerate(image_files):
            # 读取图像（使用read_image支持中文路径）
            try:
                img = read_image(img_file)
            except Exception as e:
                self._print(f"⚠️  无法读取图像: {img_file} - {e}")
                continue
            
            # ========== 完全符合run_test.py的检测逻辑 ==========
            # 方法1: 尝试通用圆形网格检测
            success, centers, ellipses = FindCirclesGridPattern(img, pat_size)
            detection_method = None
            
            if not success:
                # 尝试不同的阈值
                for thresh in range(10, 250, 10):
                    success, centers, ellipses = FindCirclesGridPattern(img, pat_size, thresh)
                    if success:
                        detection_method = f"FindCirclesGridPattern (阈值={thresh})"
                        break
            else:
                detection_method = "FindCirclesGridPattern (自动阈值)"
            
            # 如果第一种方法失败，尝试Halcon方法
            if not success:
                success, centers, ellipses = FindHalconCalibBoard(img, pat_size)
                
                if not success:
                    for thresh in range(10, 250, 10):
                        success, centers, ellipses = FindHalconCalibBoard(img, pat_size, thresh)
                        if success:
                            detection_method = f"FindHalconCalibBoard (阈值={thresh})"
                            break
                else:
                    detection_method = "FindHalconCalibBoard (自动阈值)"
            
            if success:
                # 转换为numpy数组格式 (N, 2)
                centers_array = np.array(centers, dtype=np.float32).reshape(-1, 2)
                
                # 获取世界坐标点
                world_points = self.pattern_board.generate_world_points_grid()
                
                points_pixel.append(centers_array)
                points_world.append(world_points)
                
                self._print(f"   ✓ 图像 {idx+1}/{len(image_files)}: 检测到 {len(centers)} 个标记点 [{detection_method}]")
            else:
                self._print(f"   ✗ 图像 {idx+1}/{len(image_files)}: 检测失败")
        
        if len(points_pixel) == 0:
            raise ValueError("所有图像的角点检测都失败了！")
        
        self._print(f"   📊 成功检测 {len(points_pixel)}/{len(image_files)} 张图像")
        
        return points_world, points_pixel
    
    def _step1_solve_H_A_Ks(self, points_world, points_pixel):
        """Step 1: 求解H矩阵、内参A和截断外参Ks"""
        
        def calculate_homography_matrix(points_pixel, points_world):
            """
            使用cv2.findHomography，method0代表DLT算法，计算单应性矩阵H

            参数:
            points_pixel: 所有照片的像素坐标点 shape: (n_images, n_points_per_image, 2)
            points_world: 所有照片的世界坐标点 shape: (n_images, n_points_per_image, 2) (假设Z=0)

            返回:
            H_matrices: 每张图片的单应性矩阵列表
            """
            
            H_matrices = []

            for i in range(len(points_pixel)):
                pixel_points = points_pixel[i]  # 当前图像的像素点
                world_points = points_world[i]  # 当前图像的世界坐标点
                
                # 只取世界坐标的前两个分量（X, Y），因为Z=0
                src_pts = np.array(world_points)[:, :2].astype(np.float32)
                dst_pts = np.array(pixel_points).astype(np.float32)

                # cv2.findHomography 要求输入为 (N,2) 的float32数组
                H, _ = cv2.findHomography(src_pts, dst_pts, method=0)  # 直接线性变换

                if H is None:
                    raise ValueError(f"图像 {i} 的单应性矩阵计算失败！")

                # 归一化（通常让H[2,2] = 1）
                if H[2, 2] != 0:
                    H = H / H[2, 2]

                H_matrices.append(H)

            return H_matrices
        
        def calculate_intrinsic_matrix(H_matrices):
            """
            根据一系列单应性矩阵，计算并返回远心相机的内参 alpha, beta, gamma。
            该实现基于Chen等(2014)和Li等(2024)论文中描述的线性标定方法。

            :param H_matrices: 一个列表，其中每个元素都是一个3x3的单应性矩阵(Numpy array)。
            :return: 一个元组 (alpha, beta, gamma)，包含了计算出的三个内参。
            :raises ValueError: 如果输入的单应性矩阵数量少于4个，或者在计算中出现数学错误（如对负数开方）。
            """
            
            num_H = len(H_matrices)
            if num_H < 4:
                raise ValueError(f"至少需要4个单应性矩阵进行求解，但只提供了 {num_H} 个。")

            # 初始化G矩阵和w向量
            G = np.zeros((num_H, 4))
            w = np.zeros((num_H, 1))

            # 步骤1: 构建线性方程组 GL = w
            for i, H in enumerate(H_matrices):
                # 提取H矩阵的元素用于计算
                h11, h12, _ = H[0]
                h21, h22, _ = H[1]
                
                # 根据公式(14)构建G矩阵的一行
                g1 = 1.0
                g2 = -(h21**2 + h22**2)
                g3 = -(h11**2 + h12**2)
                g4 = 2 * (h11 * h21 + h12 * h22)
                
                G[i] = [g1, g2, g3, g4]

                # 构建w向量的一个元素
                w[i] = -(h11 * h22 - h12 * h21)**2

            # 步骤2: 使用最小二乘法求解线性系统
            try:
                L, _, _, _ = np.linalg.lstsq(G, w, rcond=None)
                l1, l2, l3, l4 = L.flatten()
            except np.linalg.LinAlgError as e:
                raise ValueError(f"求解线性方程组失败，请检查H矩阵是否有效。错误: {e}")

            # 步骤3: 从L中分解出alpha, beta, gamma
            # 计算 α = sqrt((l2*l3 - l4²)/l3)
            alpha_numerator = l2 * l3 - l4**2
            if alpha_numerator <= 0:
                raise ValueError(f"alpha分子为非正数({alpha_numerator:.4f})，无法计算alpha。")
            
            alpha = np.sqrt(alpha_numerator / l3)
            
            # 计算 β = sqrt(l3)
            if l3 <= 0:
                raise ValueError(f"l3值为非正数({l3:.4f})，无法计算beta。这通常意味着输入数据有误或存在退化配置。")
            
            beta = np.sqrt(l3)
            gamma = l4/beta
            A_matrix = np.array([[alpha, gamma, 0],
                                 [0, beta, 0],
                                 [0, 0, 1]])
            return A_matrix
        
        def calculate_truncated_extrinsic_matrix(A_matrix, H_matrices):
            """
            根据相机的内参矩阵A和一系列单应性矩阵，
            计算并返回每个姿态对应的截断外参矩阵K_s。
            
            此函数依据公式 Ks = A⁻¹ * H 实现。

            :param A_matrix: 相机内参矩阵 (3x3 Numpy array)
            :param H_matrices: 一个列表，其中每个元素都是一个3x3的单应性矩阵 (Numpy array)。
            :return: 一个列表，其中每个元素都是一个3x3的截断外参矩阵K_s。
            :raises ValueError: 如果内参矩阵A是奇异的（不可逆）。
            """
            
            # 计算内参矩阵A的逆矩阵 A⁻¹
            try:
                A_inv = np.linalg.inv(A_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("内参矩阵A是奇异矩阵，无法求逆。")

            Ks_list = []
            
            # 遍历每一个单应性矩阵H，并计算对应的K_s
            for H in H_matrices:
                # 根据公式计算 K_s = A⁻¹ * H
                Ks = A_inv @ H
                Ks_list.append(Ks)
            return Ks_list
        
        H_matrices = calculate_homography_matrix(points_pixel, points_world)
        A_matrix = calculate_intrinsic_matrix(H_matrices)
        Ks_list = calculate_truncated_extrinsic_matrix(A_matrix, H_matrices)
        
        return H_matrices, A_matrix, Ks_list
    
    def _step2_solve_r31_r32_magnitudes(self, Ks_list):
        """Step 2: 依据正交性和模长相等约束，求解r31和r32的绝对值"""
        from scipy.optimize import fsolve
        
        magnitudes_list = []

        for Ks in Ks_list:
            # 1. 从Ks矩阵中提取已知的外参元素
            r11, r21, _ = Ks[:, 0]
            r12, r22, _ = Ks[:, 1]

            # 2. 定义基于新约束的方程组
            # 正交性约束: r11*r12 + r21*r22 + r31*r32 = 0
            # 模长相等约束: r11^2+r21^2+r31^2 = r12^2+r22^2+r32^2
            def equations(p):
                eq1 = r11*r12 + r21*r22 + p[0]*p[1]
                eq2 = (r11**2 + r21**2 + p[0]**2) - (r12**2 + r22**2 + p[1]**2)
                return (eq1, eq2)

            # 3. 求解方程组
            r31_solution, r32_solution = fsolve(equations, (0.01,0.01))

            # 4. 提取解的绝对值
            abs_r31 = np.abs(r31_solution)
            abs_r32 = np.abs(r32_solution)

            magnitudes_list.append((abs_r31, abs_r32))
        
        return magnitudes_list

    def _step3_determine_r31_r32_signs(self, points_world, points_pixel):
        """Step 3: 按照Li Hai等人(2024)论文中所描述的，使用针孔模型确定r31和r32的符号"""
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            points_world,
            points_pixel,
            self.image_size,
            None,
            None
        )
        
        if not ret:
            raise RuntimeError("针孔模型标定失败，无法获取参考符号")
        else:
            # 详细调试信息仅在verbose模式下输出
            self._print(f"重投影误差为：{ret}，内参矩阵为：{mtx}，畸变系数为：{dist}，旋转向量rvecs为：{rvecs}，平移向量tvecs为：{tvecs}")
            
        reference_signs_list = []
        for rvec in rvecs:
            # 将旋转向量转换为旋转矩阵，获取针孔模型的旋转矩阵作为符号参考
            R_pinhole, _ = cv2.Rodrigues(rvec)
            
            sign_r31 = np.sign(R_pinhole[2, 0])
            sign_r32 = np.sign(R_pinhole[2, 1])
            
            if sign_r31 == 0:
                sign_r31 = 1.0
            if sign_r32 == 0:
                sign_r32 = 1.0
            
            reference_signs_list.append((sign_r31, sign_r32))
        
        return reference_signs_list
    
    def _step4_construct_rotation_matrices(self, Ks_list, magnitudes_list, signs_list):
        """Step 4: 通过向量归一化和交叉积求出完整的旋转矩阵R，并利用SVD估计出最优的旋转矩阵Ropt"""
        R_list = []
        for Ks, (abs_r31, abs_r32), (sign_r31, sign_r32) in zip(Ks_list, magnitudes_list, signs_list):
            # 计算正确的r31和r32
            correct_r31 = abs_r31 * sign_r31
            correct_r32 = abs_r32 * sign_r32
            
            r11, r21, _ = Ks[:, 0]
            r12, r22, _ = Ks[:, 1]
            
            rx = np.array([r11, r21, correct_r31])
            ry = np.array([r12, r22, correct_r32])
            
            # 通过叉乘得到rz
            rz = np.cross(rx, ry)
            
            # 构造旋转矩阵R，其列为rx, ry, rz
            R_approx = np.column_stack([rx, ry, rz])
            
            # 使用SVD方法对R进行强制正交化
            U, _, Vt = np.linalg.svd(R_approx)
            
            # 强制正交化：R_ortho = U @ Vt
            R_ortho = U @ Vt
            
            R_list.append(R_ortho)
        
        return R_list
    
    def _step5_global_optimization(self, points_world, points_pixel, H_matrices, A_matrix, Ks_list, R_list):
        """
        Step 5: 引入畸变模型并执行全局优化
        
        使用 Levenberg-Marquardt 算法最小化重投影误差：
        Σ Σ ||p_ij - p̃(A, K_s^(i), k1, k2, P_j)||²
        
        参数:
            points_world: 世界坐标点列表
            points_pixel: 像素坐标点列表
            H_matrices: 单应性矩阵列表（未使用）
            A_matrix: 初始内参 (alpha, beta, gamma)
            Ks_list: 截断外参矩阵列表
            R_list: 旋转矩阵列表
        
        返回:
            优化后的参数
        """
        from scipy.optimize import least_squares
        
        num_images = len(points_world)
        
        # ========== 1. 准备初始参数 ==========
        # 内参初始值
        alpha = A_matrix[0, 0]
        beta = A_matrix[1, 1]
        gamma = A_matrix[0, 1]
        
        # 畸变系数初始值
        k1, k2 = 0.0, 0.0
        
        # 将旋转矩阵转为旋转向量（Rodrigues变换）
        rvecs = [cv2.Rodrigues(R)[0] for R in R_list]
        
        # 从 Ks_list 提取平移向量（最后一列的前两个值）
        tvecs = [Ks[:2, 2] for Ks in Ks_list]
        
        # 打包所有参数为一维向量
        params_vec = np.concatenate((
            [alpha, beta, gamma, k1, k2],
            np.array(rvecs).flatten(),
            np.array(tvecs).flatten()
        ))
        
        # ========== 2. 定义残差函数 ==========
        def residuals(params):
            """
            计算残差：observed - predicted
            符合公式(1)的远心相机投影模型
            """
            # 解包参数
            alpha, beta, gamma = params[0:3]
            k1, k2 = params[3:5]
            
            rvecs_flat = params[5 : 5 + num_images*3]
            tvecs_flat = params[5 + num_images*3 :]
            
            rvecs = rvecs_flat.reshape((num_images, 3))
            tvecs = tvecs_flat.reshape((num_images, 2))
            
            errors = []
            
            # 构建内参矩阵 ξ (符合图2)
            M_int = np.array([
                [alpha, gamma, 0, 0],
                [0,     beta,  0, 0],
                [0,     0,     0, 1]
            ])
            
            # 遍历每张图像
            for i in range(num_images):
                world_pts = points_world[i]  # (N, 3)
                pixel_pts_observed = points_pixel[i].reshape(-1, 2)  # (N, 2)
                
                # 构建外参矩阵 κ
                R, _ = cv2.Rodrigues(rvecs[i])
                tx, ty = tvecs[i]
                T = np.array([tx, ty, 0]).reshape(3, 1)  # tz = 0（远心相机特性）
                M_ext = np.vstack((np.hstack((R, T)), [0, 0, 0, 1]))  # 4x4
                
                # 将世界坐标转为齐次坐标 (4, N)
                num_pts = world_pts.shape[0]
                P_w = np.vstack((world_pts.T, np.ones(num_pts)))
                
                # 按照公式(1)投影：[u; v; 1] = ξ · κ · [Xw; Yw; Zw; 1]
                proj_homo = M_int @ M_ext @ P_w
                
                # 理想像素坐标（无畸变）
                u_ideal = proj_homo[0, :]
                v_ideal = proj_homo[1, :]
                
                # 应用径向畸变模型
                r_sq = u_ideal**2 + v_ideal**2
                distortion_factor = 1 + k1 * r_sq + k2 * r_sq**2
                u_pred = u_ideal * distortion_factor
                v_pred = v_ideal * distortion_factor
                
                pixel_pts_predicted = np.vstack((u_pred, v_pred)).T
                
                # 计算残差向量
                error = (pixel_pts_observed - pixel_pts_predicted).flatten()
                errors.extend(error)
            
            return np.array(errors)
        
        # ========== 3. 执行 Levenberg-Marquardt 优化 ==========
        self._print("   正在执行 Levenberg-Marquardt 优化...")
        # scipy的verbose参数：0=静默，1=显示，2=详细
        scipy_verbose = 1 if self.verbose else 0
        result = least_squares(
            residuals, 
            params_vec, 
            method='lm',  # Levenberg-Marquardt 算法
            verbose=scipy_verbose
        )
        
        # ========== 4. 解包优化后的参数 ==========
        optimized_params = result.x
        
        alpha_opt, beta_opt, gamma_opt = optimized_params[0:3]
        k1_opt, k2_opt = optimized_params[3:5]
        
        rvecs_opt = optimized_params[5:5+num_images*3].reshape((num_images, 3))
        tvecs_opt = optimized_params[5+num_images*3:].reshape((num_images, 2))
        
        # 将旋转向量转回旋转矩阵
        R_list_opt = [cv2.Rodrigues(rvec)[0] for rvec in rvecs_opt]
        
        # ========== 5. 更新参数容器 ==========
        self.param_holder.set_intrinsic(
            alpha=float(alpha_opt),
            beta=float(beta_opt),
            gamma=float(gamma_opt),
            k1=float(k1_opt),
            k2=float(k2_opt)
        )
        
        # 更新外参（使用第一组作为代表）
        self.param_holder.set_extrinsic(
            R=R_list_opt[0],
            T=np.array([tvecs_opt[0][0], tvecs_opt[0][1], 0])
        )
        
        self._print(f"   📐 优化后内参: alpha={alpha_opt:.4f}, beta={beta_opt:.4f}, gamma={gamma_opt:.4f}")
        self._print(f"   🔍 优化后畸变: k1={k1_opt:.6e}, k2={k2_opt:.6e}")
        
        return alpha_opt, beta_opt, gamma_opt, k1_opt, k2_opt, R_list_opt, tvecs_opt

class DualCameraCalibrator:
    def __init__(self, left_camera:TelecentricCalibrator, right_camera:TelecentricCalibrator, binocular_image_dir:str, 
                 left_image_name:str = "0_left.bmp", right_image_name:str = "0_right.bmp", verbose:bool = True):
        """
        初始化双目标定器
        
        参数:
            left_camera: 左相机标定器
            right_camera: 右相机标定器
            binocular_image_dir: 双目图像目录
            left_image_name: 左相机图像文件名（默认: "0_left.bmp"）
            right_image_name: 右相机图像文件名（默认: "0_right.bmp"）
            verbose: 是否显示详细输出（默认True）
        """
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.binocular_image_dir = binocular_image_dir
        self.left_image_name = left_image_name
        self.right_image_name = right_image_name
        self.verbose = verbose
    
    def _print(self, *args, **kwargs):
        """条件输出方法，仅在verbose=True时输出"""
        if self.verbose:
            print(*args, **kwargs)
        
    def undistort_points(self, points_distorted, k1, k2, max_iter=20, tolerance=1e-6):
        """
        对畸变的像素坐标进行去畸变处理
        
        畸变模型: p_distorted = p_ideal * (1 + k1*r^2 + k2*r^4)
        其中 r^2 = u_ideal^2 + v_ideal^2
        
        参数:
            points_distorted: 带畸变的像素坐标 (N, 2)
            k1, k2: 畸变系数
            max_iter: 最大迭代次数
            tolerance: 收敛容差
        
        返回:
            points_ideal: 去畸变后的理想像素坐标 (N, 2)
        """
        points_ideal = points_distorted.copy()
        
        for iteration in range(max_iter):
            r_sq = points_ideal[:, 0]**2 + points_ideal[:, 1]**2
            distortion_factor = 1 + k1 * r_sq + k2 * r_sq**2
            
            # 根据当前理想坐标计算应该产生的畸变坐标
            points_predicted = points_ideal * distortion_factor[:, np.newaxis]
            
            # 计算误差
            error = points_distorted - points_predicted
            
            # 更新理想坐标
            points_ideal = points_ideal + error / distortion_factor[:, np.newaxis]
            
            # 检查收敛
            if np.max(np.abs(error)) < tolerance:
                break
        
        return points_ideal
    
    def dual_camera_calibration(self):
        # 利用已经标定好的单相机参数，标定双相机参数
        # 读取左右相机的共同图像（使用配置的文件名）
        left_image_path = os.path.join(self.binocular_image_dir, self.left_image_name)
        right_image_path = os.path.join(self.binocular_image_dir, self.right_image_name)
        
        # 检测左右图像的角点
        self._print("\n===== 双目标定：检测对应点 =====")
        left_image = read_image(left_image_path)
        right_image = read_image(right_image_path)
        
        # 使用 pattern 包检测标定板圆点
        # FindHalconCalibBoard 和 FindCirclesGridPattern 已在文件开头导入
        board_params = self.left_camera.param_holder.get_board_dict()
        pat_size = board_params['dot_array_shape']
        
        # ========== 检测左图像 ==========
        success_left, centers_left, ellipses_left = FindCirclesGridPattern(left_image, pat_size)
        
        if not success_left:
            for thresh in range(10, 250, 10):
                success_left, centers_left, ellipses_left = FindCirclesGridPattern(left_image, pat_size, thresh)
                if success_left:
                    break
        
        if not success_left:
            success_left, centers_left, ellipses_left = FindHalconCalibBoard(left_image, pat_size)
            
            if not success_left:
                for thresh in range(10, 250, 10):
                    success_left, centers_left, ellipses_left = FindHalconCalibBoard(left_image, pat_size, thresh)
                    if success_left:
                        break
        
        # ========== 检测右图像 ==========
        success_right, centers_right, ellipses_right = FindCirclesGridPattern(right_image, pat_size)
        
        if not success_right:
            for thresh in range(10, 250, 10):
                success_right, centers_right, ellipses_right = FindCirclesGridPattern(right_image, pat_size, thresh)
                if success_right:
                    break
        
        if not success_right:
            success_right, centers_right, ellipses_right = FindHalconCalibBoard(right_image, pat_size)
            
            if not success_right:
                for thresh in range(10, 250, 10):
                    success_right, centers_right, ellipses_right = FindHalconCalibBoard(right_image, pat_size, thresh)
                    if success_right:
                        break
        
        if not success_left or not success_right:
            raise ValueError("角点检测失败！")
        
        points_pixel_left = np.array(centers_left, dtype=np.float32).reshape(-1, 2)
        points_pixel_right = np.array(centers_right, dtype=np.float32).reshape(-1, 2)
        
        self._print(f"检测到 {len(points_pixel_left)} 对对应点")
        
        # 获取外参
        left_rvec, left_tvec, left_R = self.left_camera.known_intrinsic_calibration_single_image(left_image_path)
        right_rvec, right_tvec, right_R = self.right_camera.known_intrinsic_calibration_single_image(right_image_path)
        
        ## 利用左右相机的内外参矩阵，得到最终M矩阵
        Mat_intrinsic_left = np.array([
            [self.left_camera.param_holder.alpha, self.left_camera.param_holder.gamma, 0, 0],
            [0, self.left_camera.param_holder.beta, 0, 0],
            [0, 0, 0, 1]
        ])
        Mat_intrinsic_right = np.array([
            [self.right_camera.param_holder.alpha, self.right_camera.param_holder.gamma, 0, 0],
            [0, self.right_camera.param_holder.beta, 0, 0],
            [0, 0, 0, 1]
        ])
        Mat_extrinsic_left = np.array([
            [left_R[0, 0], left_R[0, 1], left_R[0, 2], left_tvec[0]],
            [left_R[1, 0], left_R[1, 1], left_R[1, 2], left_tvec[1]],
            [left_R[2, 0], left_R[2, 1], left_R[2, 2], 0],
            [0, 0, 0, 1]
        ])
        Mat_extrinsic_right = np.array([
            [right_R[0, 0], right_R[0, 1], right_R[0, 2], right_tvec[0]],
            [right_R[1, 0], right_R[1, 1], right_R[1, 2], right_tvec[1]],
            [right_R[2, 0], right_R[2, 1], right_R[2, 2], 0],
            [0, 0, 0, 1]
        ])
        
        Mat_left = Mat_intrinsic_left @ Mat_extrinsic_left
        Mat_right = Mat_intrinsic_right @ Mat_extrinsic_right

        self._print(f"\n左相机M矩阵:\n{Mat_left}")
        self._print(f"\n右相机M矩阵:\n{Mat_right}")
        
        # 去畸变处理
        self._print("\n===== 去畸变处理 =====")
        k1_left = self.left_camera.param_holder.k1
        k2_left = self.left_camera.param_holder.k2
        k1_right = self.right_camera.param_holder.k1
        k2_right = self.right_camera.param_holder.k2
        
        self._print(f"左相机畸变系数: k1={k1_left:.6e}, k2={k2_left:.6e}")
        self._print(f"右相机畸变系数: k1={k1_right:.6e}, k2={k2_right:.6e}")
        
        points_ideal_left = self.undistort_points(points_pixel_left, k1_left, k2_left)
        points_ideal_right = self.undistort_points(points_pixel_right, k1_right, k2_right)
        
        # 获取理论世界坐标（用于对比）
        points_world_theory = self.left_camera.pattern_board.generate_world_points_grid()
        
        # 从M矩阵中提取前两行（对应u和v方程）
        M_left = Mat_left[:2, :]  # 2x4
        M_right = Mat_right[:2, :]  # 2x4
        
        # ========== 对比实验1：不带畸变校正的重建 ==========
        self._print("\n" + "="*60)
        self._print("对比实验1：使用原始像素坐标（不带畸变校正）")
        self._print("="*60)
        
        reconstructed_points_distorted = []
        
        for i in range(len(points_pixel_left)):
            p = points_pixel_left[i]  # 带畸变的原始像素坐标
            p_prime = points_pixel_right[i]
            
            # 构建线性方程组
            A = np.vstack([
                M_left[:, :3],
                M_right[:, :3]
            ])
            
            b = np.hstack([
                p - M_left[:, 3],
                p_prime - M_right[:, 3]
            ])
            
            P, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            reconstructed_points_distorted.append(P)
        
        reconstructed_points_distorted = np.array(reconstructed_points_distorted)
        errors_distorted = reconstructed_points_distorted - points_world_theory
        
        self._print("\n【不带畸变校正的重建精度】")
        self._print(f"X方向 - 均值: {np.mean(errors_distorted[:, 0]):.4f} mm, 标准差: {np.std(errors_distorted[:, 0]):.4f} mm, 最大值: {np.max(np.abs(errors_distorted[:, 0])):.4f} mm")
        self._print(f"Y方向 - 均值: {np.mean(errors_distorted[:, 1]):.4f} mm, 标准差: {np.std(errors_distorted[:, 1]):.4f} mm, 最大值: {np.max(np.abs(errors_distorted[:, 1])):.4f} mm")
        self._print(f"Z方向 - 均值: {np.mean(errors_distorted[:, 2]):.4f} mm, 标准差: {np.std(errors_distorted[:, 2]):.4f} mm, 最大值: {np.max(np.abs(errors_distorted[:, 2])):.4f} mm")
        self._print(f"整体RMS误差: {np.sqrt(np.mean(errors_distorted**2)):.4f} mm")
        
        # ========== 对比实验2：带畸变校正的重建 ==========
        self._print("\n" + "="*60)
        self._print("对比实验2：使用去畸变后的像素坐标（带畸变校正）")
        self._print("="*60)
        
        reconstructed_points_undistorted = []
        
        for i in range(len(points_ideal_left)):
            p = points_ideal_left[i]  # 去畸变后的理想像素坐标
            p_prime = points_ideal_right[i]
            
            # 构建线性方程组
            A = np.vstack([
                M_left[:, :3],
                M_right[:, :3]
            ])
            
            b = np.hstack([
                p - M_left[:, 3],
                p_prime - M_right[:, 3]
            ])
            
            P, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            reconstructed_points_undistorted.append(P)
        
        reconstructed_points_undistorted = np.array(reconstructed_points_undistorted)
        errors_undistorted = reconstructed_points_undistorted - points_world_theory
        
        self._print("\n【带畸变校正的重建精度】")
        self._print(f"X方向 - 均值: {np.mean(errors_undistorted[:, 0]):.4f} mm, 标准差: {np.std(errors_undistorted[:, 0]):.4f} mm, 最大值: {np.max(np.abs(errors_undistorted[:, 0])):.4f} mm")
        self._print(f"Y方向 - 均值: {np.mean(errors_undistorted[:, 1]):.4f} mm, 标准差: {np.std(errors_undistorted[:, 1]):.4f} mm, 最大值: {np.max(np.abs(errors_undistorted[:, 1])):.4f} mm")
        self._print(f"Z方向 - 均值: {np.mean(errors_undistorted[:, 2]):.4f} mm, 标准差: {np.std(errors_undistorted[:, 2]):.4f} mm, 最大值: {np.max(np.abs(errors_undistorted[:, 2])):.4f} mm")
        self._print(f"整体RMS误差: {np.sqrt(np.mean(errors_undistorted**2)):.4f} mm")
        
        # ========== 对比总结 ==========
        self._print("\n" + "="*60)
        self._print("📊 畸变校正效果对比总结")
        self._print("="*60)
        
        rms_distorted = np.sqrt(np.mean(errors_distorted**2))
        rms_undistorted = np.sqrt(np.mean(errors_undistorted**2))
        improvement = (rms_distorted - rms_undistorted) / rms_distorted * 100
        
        self._print(f"\n不带畸变校正 RMS误差: {rms_distorted:.4f} mm")
        self._print(f"带畸变校正 RMS误差:   {rms_undistorted:.4f} mm")
        self._print(f"精度提升: {improvement:.2f}%")
        
        return Mat_left, Mat_right, reconstructed_points_undistorted, points_world_theory, {
            'distorted': reconstructed_points_distorted,
            'undistorted': reconstructed_points_undistorted,
            'errors_distorted': errors_distorted,
            'errors_undistorted': errors_undistorted,
            'points_pixel_left': points_pixel_left,
            'points_pixel_right': points_pixel_right
        }
    
    def save_fundamental_matrix_json(self, points_left, points_right, output_path="fundamentalMatrix.json"):
        """
        计算并保存基础矩阵到JSON文件
        
        参数:
            points_left: 左相机像素坐标点
            points_right: 右相机像素坐标点
            output_path: 输出文件路径
        """
        # 使用RANSAC方法计算基础矩阵
        F_matrix, mask = cv2.findFundamentalMat(
            points_left, 
            points_right, 
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99
        )
        
        if F_matrix is None:
            self._print("   ⚠️  基础矩阵计算失败")
            return None
        
        # 统计内点数量
        inliers = np.sum(mask)
        total_points = len(points_left)
        
        # 准备JSON数据
        data = {
            "fundamental_matrix": F_matrix.tolist(),
            "description": "使用多组同名点计算得到的基础矩阵",
            "total_points": int(total_points),
            "inliers": int(inliers),
            "method": "RANSAC"
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        self._print(f"   ✅ 基础矩阵已保存到: {output_path}")
        self._print(f"   总点数: {total_points}, 内点数: {inliers} ({inliers/total_points*100:.1f}%)")
        
        return F_matrix
    
    def save_reconstruction_errors_json(self, points_theory, points_reconstructed_distorted, 
                                       points_reconstructed_undistorted, output_path="reconstruction_errors.json"):
        """
        保存每个点的重建误差统计到JSON文件
        
        参数:
            points_theory: 理论标记点坐标 (N, 3)
            points_reconstructed_distorted: 不带畸变校正的重建点 (N, 3)
            points_reconstructed_undistorted: 带畸变校正的重建点 (N, 3)
            output_path: 输出文件路径
        """
        # 计算误差
        errors_distorted = points_reconstructed_distorted - points_theory
        errors_undistorted = points_reconstructed_undistorted - points_theory
        
        # 构建每个点的详细信息
        points_details = []
        for i in range(len(points_theory)):
            point_info = {
                "point_index": i,
                "theory": {
                    "X": float(points_theory[i, 0]),
                    "Y": float(points_theory[i, 1]),
                    "Z": float(points_theory[i, 2])
                },
                "reconstructed_without_distortion_correction": {
                    "X": float(points_reconstructed_distorted[i, 0]),
                    "Y": float(points_reconstructed_distorted[i, 1]),
                    "Z": float(points_reconstructed_distorted[i, 2])
                },
                "reconstructed_with_distortion_correction": {
                    "X": float(points_reconstructed_undistorted[i, 0]),
                    "Y": float(points_reconstructed_undistorted[i, 1]),
                    "Z": float(points_reconstructed_undistorted[i, 2])
                },
                "error_without_correction": {
                    "dX": float(errors_distorted[i, 0]),
                    "dY": float(errors_distorted[i, 1]),
                    "dZ": float(errors_distorted[i, 2]),
                    "norm": float(np.linalg.norm(errors_distorted[i]))
                },
                "error_with_correction": {
                    "dX": float(errors_undistorted[i, 0]),
                    "dY": float(errors_undistorted[i, 1]),
                    "dZ": float(errors_undistorted[i, 2]),
                    "norm": float(np.linalg.norm(errors_undistorted[i]))
                }
            }
            points_details.append(point_info)
        
        # 计算统计信息
        statistics = {
            "without_distortion_correction": {
                "total_points": len(points_theory),
                "mean_error": {
                    "X": float(np.mean(errors_distorted[:, 0])),
                    "Y": float(np.mean(errors_distorted[:, 1])),
                    "Z": float(np.mean(errors_distorted[:, 2]))
                },
                "std_error": {
                    "X": float(np.std(errors_distorted[:, 0])),
                    "Y": float(np.std(errors_distorted[:, 1])),
                    "Z": float(np.std(errors_distorted[:, 2]))
                },
                "max_abs_error": {
                    "X": float(np.max(np.abs(errors_distorted[:, 0]))),
                    "Y": float(np.max(np.abs(errors_distorted[:, 1]))),
                    "Z": float(np.max(np.abs(errors_distorted[:, 2])))
                },
                "rms_error": float(np.sqrt(np.mean(errors_distorted**2)))
            },
            "with_distortion_correction": {
                "total_points": len(points_theory),
                "mean_error": {
                    "X": float(np.mean(errors_undistorted[:, 0])),
                    "Y": float(np.mean(errors_undistorted[:, 1])),
                    "Z": float(np.mean(errors_undistorted[:, 2]))
                },
                "std_error": {
                    "X": float(np.std(errors_undistorted[:, 0])),
                    "Y": float(np.std(errors_undistorted[:, 1])),
                    "Z": float(np.std(errors_undistorted[:, 2]))
                },
                "max_abs_error": {
                    "X": float(np.max(np.abs(errors_undistorted[:, 0]))),
                    "Y": float(np.max(np.abs(errors_undistorted[:, 1]))),
                    "Z": float(np.max(np.abs(errors_undistorted[:, 2])))
                },
                "rms_error": float(np.sqrt(np.mean(errors_undistorted**2)))
            }
        }
        
        # 计算改善率
        rms_distorted = statistics["without_distortion_correction"]["rms_error"]
        rms_undistorted = statistics["with_distortion_correction"]["rms_error"]
        improvement = (rms_distorted - rms_undistorted) / rms_distorted * 100 if rms_distorted != 0 else 0
        
        # 构建完整输出
        output_data = {
            "description": "三维重建误差统计（理论标记点 vs 重建点）",
            "unit": "mm",
            "statistics": statistics,
            "improvement_percentage": float(improvement),
            "points_details": points_details
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        self._print(f"   ✅ 重建误差统计已保存到: {output_path}")
        self._print(f"   总点数: {len(points_theory)}")
        self._print(f"   不带校正 RMS误差: {rms_distorted:.4f} mm")
        self._print(f"   带校正 RMS误差: {rms_undistorted:.4f} mm")
        self._print(f"   精度提升: {improvement:.2f}%")
        
        return output_data

