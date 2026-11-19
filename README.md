# 远心相机标定项目

## 项目简介

这是一个用于远心相机标定的Python项目，支持单目和双目标定。项目实现了基于平面标定板的远心相机标定算法，主要参考了Zhong Chen (2014) 和 Hai Li (2024) 的论文方法。

## 主要特性

- ✅ **单目相机标定**：支持远心相机的内参、外参和畸变系数标定
- ✅ **双目相机标定**：支持双远心相机系统的标定和三维重建
- ✅ **配置文件支持**：通过YAML配置文件管理所有参数，无需修改代码
- ✅ **详细输出控制**：支持verbose参数控制输出详细程度
- ✅ **中文路径支持**：完全支持中文路径的图像读取和保存
- ✅ **模块化设计**：清晰的代码结构，便于维护和扩展

## 安装说明

### 环境要求

- Python >= 3.7
- Windows / Linux / macOS

### 安装步骤

1. 克隆或下载本项目到本地

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

3. 准备配置文件：
```bash
# 复制配置文件模板
cp config.example.yaml config.yaml

# 编辑config.yaml，设置你的图像路径和标定板参数
```

## 快速开始

### 1. 配置设置

编辑 `config.yaml` 文件，设置以下参数：

- **路径配置**：设置标定图像目录和输出目录
- **标定板参数**：设置圆点阵列形状、直径和间距
- **输出设置**：设置是否显示详细输出和输出文件名

### 2. 单相机标定

运行单相机标定示例：

```bash
python examples/example_single_camera.py
```

### 3. 双相机标定

运行双相机标定示例：

```bash
python examples/example_dual_camera.py
```

## 详细使用说明

### 单目标定流程

1. **准备标定图像**：拍摄多张（建议10张以上）不同姿态的标定板图像
2. **配置参数**：在 `config.yaml` 中设置图像目录和标定板参数
3. **运行标定**：执行 `example_single_camera.py`
4. **获取结果**：标定结果保存在输出目录的JSON文件中

### 双目标定流程

1. **单相机标定**：分别对左右相机进行单目标定
2. **准备双目图像**：拍摄左右相机同步采集的标定板图像
3. **配置参数**：在 `config.yaml` 中设置双目图像目录和文件名
4. **运行双目标定**：执行 `example_dual_camera.py`
5. **获取结果**：获得基础矩阵、重建误差等结果

### 配置文件说明

配置文件 `config.yaml` 包含以下主要部分：

```yaml
paths:
  left_camera_images: "image/left"      # 左相机图像目录
  right_camera_images: "image/right"    # 右相机图像目录
  binocular_images: "image/binocular"   # 双目图像目录
  output_dir: "output"                  # 输出目录

calibration_board:
  dot_array_shape: [7, 7]               # 圆点阵列 (行, 列)
  dot_diameter: 0.1875                  # 圆点直径 (mm)
  dot_spacing: 0.375                    # 圆心间距 (mm)

binocular:
  left_image_name: "0_left.bmp"         # 左相机图像文件名
  right_image_name: "0_right.bmp"       # 右相机图像文件名

output:
  verbose: true                          # 是否显示详细输出
  left_camera_output: "leftCKmatrix.json"
  right_camera_output: "rightCKmatrix.json"
```

## 项目结构

```
telecentric-camera-calibration/
├── README.md                      # 本文件
├── LICENSE                        # BSD-2-Clause许可证
├── requirements.txt               # 依赖列表
├── .gitignore                    # Git忽略规则
├── config.example.yaml            # 配置文件模板
├── docs/                          # 文档目录
│   ├── 数学推导.md                # 数学原理详细推导
│   └── images/                    # 文档图片
├── telecentric_calib/             # 主代码包
│   ├── __init__.py
│   ├── config.py                 # 配置管理模块
│   ├── calibrator.py             # 标定器核心模块
│   ├── detector.py               # 标定侧检测入口（标定板与回退策略）
│   ├── pattern/                  # 圆点检测算法
│   │   ├── ellipse.py            # 亚像素椭圆拟合
│   │   ├── geometry.py           # 角点、聚类、排序
│   │   ├── circles.py            # 普通圆网格
│   │   └── halcon_board.py       # HALCON 标定板
│   ├── params.py                 # 参数管理模块
│   └── utils.py                  # 工具函数模块
├── examples/                      # 示例代码
│   ├── example_single_camera.py  # 单相机标定示例
│   ├── example_dual_camera.py    # 双相机标定示例
│   └── detect/                   # 标定板检测脚本
└── ...
```

## 标定板检测

圆点检测在 `telecentric_calib.pattern` 中，是 [find_calib_pattern](https://github.com/Neucrede/find_calib_pattern) 的 Python 移植。

- **原项目**：https://github.com/Neucrede/find_calib_pattern
- **许可证**：BSD-2-Clause License
- **功能**：检测 HALCON 标定板，或提取圆形网格阵列的中心点
- **对外函数**：`FindCirclesGridPattern`、`FindHalconCalibBoard`

标定流程通过 `telecentric_calib.detector` 使用这组函数：先找普通圆网格，失败后再找 HALCON 标定板。

感谢原项目作者 Neucrede 的开源贡献！

## 数学原理

详细的数学推导过程请参考 [docs/数学推导.md](docs/数学推导.md)。

本项目实现的标定算法主要基于以下论文：

1. **Chen, Z., et al. (2014)** - "Telecentric stereo micro-vision system: Calibration method and experiments"
2. **Li, H., et al. (2024)** - "Flexible Calibration of the Telecentric Vision Systems Using Only Planar Calibration Target"

标定流程包括：
1. 单应性矩阵求解
2. 内参矩阵求解
3. 外参矩阵求解（包括旋转矩阵的完整求解）
4. 畸变模型优化

## 引用说明

如果您在研究中使用了本项目，请引用相关论文：

```bibtex
@article{chen2014telecentric,
  title={Telecentric stereo micro-vision system: Calibration method and experiments},
  author={Chen, Zhong and others},
  year={2014}
}

@article{li2024flexible,
  title={Flexible Calibration of the Telecentric Vision Systems Using Only Planar Calibration Target},
  author={Li, Hai and others},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2024}
}
```

## 参考文献

1. Chen, Z., et al. (2014). Telecentric stereo micro-vision system: Calibration method and experiments.

2. Li, H., Liao, Z., Cai, W., et al. (2024). Flexible Calibration of the Telecentric Vision Systems Using Only Planar Calibration Target. IEEE Transactions on Instrumentation and Measurement, 73, 1-10.

3. Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision. Cambridge university press.

## 测试数据

由于测试图像文件体积较大，未包含在本仓库中。用户需要自行准备标定图像（或是向作者咨询索取）：

- **单目标定**：需要多张（建议10张以上）不同姿态的标定板图像
- **双目标定**：需要左右相机分别的单目标定图像，以及同步采集的双目图像对

标定板要求：
- 使用HALCON圆形网格标定板
- 圆点阵列建议为7×7或更大
- 确保标定板在图像中清晰可见

## 许可证

本项目采用 **BSD-2-Clause License** 许可证。

```
BSD 2-Clause License

Copyright (c) 2025, YLChen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 致谢

- 感谢 [Neucrede/find_calib_pattern](https://github.com/Neucrede/find_calib_pattern) 项目提供的标定板检测算法，现位于 `telecentric_calib.pattern`
- 感谢相关论文作者提供的理论支持

## 联系方式

如有问题或建议，欢迎通过GitHub Issues联系。

---

**注意**：本项目正在持续开发中，部分功能可能需要进一步完善。欢迎贡献代码和提出建议！

