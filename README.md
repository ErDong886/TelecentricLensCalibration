[English](README.en.md) | 中文

# 远心相机标定项目

## 项目简介

这是一个用于远心相机标定的 Python 项目，支持单目和双目标定。项目实现了基于平面标定板的远心相机标定算法，主要参考 Zhong Chen (2014) 和 Hai Li (2024) 的论文。

## 主要特性

- 单目相机标定：内参、外参和径向畸变
- 双目相机标定：双远心相机的标定和三维重建
- YAML 配置：路径和标定板参数都在配置文件里
- 输出详细程度可由 `verbose` 控制
- 图像读写支持中文路径
- 业务代码集中在 `src/telecentric_calib`

## 安装说明

### 环境要求

- Python >= 3.7
- Windows / Linux / macOS

### 安装步骤

1. 克隆本项目

2. 安装依赖：

```bash
pip install -r requirements.txt
```

示例脚本会把 `src` 加入 `sys.path`，安装依赖后即可运行。若要在其他目录中 `import telecentric_calib`，再执行：

```bash
pip install -e .
```

3. 准备配置文件：

```bash
cp config.example.yaml config.yaml
```

然后编辑 `config.yaml`，填入图像路径和标定板参数。

## 快速开始

编辑 `config.yaml` 中的图像目录、圆点阵列、直径、间距和输出文件名。

单相机标定：

```bash
python examples/example_single_camera.py
```

双相机标定：

```bash
python examples/example_dual_camera.py
```

## 详细使用说明

### 单目标定流程

1. 拍摄至少 4 张不同姿态的标定板图像，建议 10 张以上
2. 在 `config.yaml` 中设置图像目录和标定板参数
3. 运行 `examples/example_single_camera.py`
4. 结果写入输出目录中的 JSON 文件

内参线性求解至少需要 4 张图像。图像按目录列举顺序读取，不另行排序。

### 双目标定流程

1. 分别完成左右相机的单目标定
2. 准备左右相机同步采集的一对标定板图像
3. 在 `config.yaml` 中设置双目图像目录和文件名
4. 运行 `examples/example_dual_camera.py`
5. 得到投影矩阵、基础矩阵和重建误差

### 配置文件说明

`config.yaml` 的主要字段：

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
  verbose: true
  left_camera_output: "leftCKmatrix.json"
  right_camera_output: "rightCKmatrix.json"
```

## 项目结构

```
telecentric-camera-calibration/
├── README.md                      # 中文说明
├── README.en.md                   # English
├── LICENSE                        # BSD-2-Clause
├── requirements.txt
├── pyproject.toml                 # src 布局安装配置
├── config.example.yaml
├── docs/
│   └── 数学推导.md
├── examples/
│   ├── example_single_camera.py
│   ├── example_dual_camera.py
│   └── detect/
└── src/telecentric_calib/
    ├── api.py                     # calibrate_mono / calibrate_stereo
    ├── config.py
    ├── params.py
    ├── board.py
    ├── io.py
    ├── detect/                    # 圆点检测
    ├── mono/                      # 单目标定
    └── stereo.py                  # 双目重建
```

## 标定板检测

圆点检测在 `telecentric_calib.detect`，是 [find_calib_pattern](https://github.com/Neucrede/find_calib_pattern) 的 Python 移植。

- 原项目：https://github.com/Neucrede/find_calib_pattern
- 许可证：BSD-2-Clause License
- 对外函数：`FindCirclesGridPattern`、`FindHalconCalibBoard`

标定先找普通圆网格，失败后再找 HALCON 标定板。对外入口是 `calibrate_mono` 和 `calibrate_stereo`。

## 数学原理

推导见 [docs/数学推导.md](docs/数学推导.md)。文档中的内参回代、线性方程组和符号求解与当前代码一致，并补充了双目重建所用的方程。

本项目主要基于：

1. Chen, Z., et al. (2014). Telecentric stereo micro-vision system: Calibration method and experiments.
2. Li, H., et al. (2024). Flexible Calibration of the Telecentric Vision Systems Using Only Planar Calibration Target.

单目标定依次求解单应矩阵、内参、完整旋转矩阵，再加入径向畸变做整体优化。

## 引用说明

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

测试图像体积较大，未包含在本仓库中。请自行准备标定图像。

- 单目标定：至少 4 张不同姿态的图像，建议 10 张以上
- 双目标定：左右相机各自的单目标定图像，以及一对同步图像
- 标定板：普通圆形网格或 HALCON 圆形标定板，阵列建议 7×7 或更大

## 许可证

本项目采用 BSD-2-Clause License，全文见 [LICENSE](LICENSE)。

## 致谢

- 感谢 [Neucrede/find_calib_pattern](https://github.com/Neucrede/find_calib_pattern) 提供标定板检测算法，现位于 `telecentric_calib.detect`
- 感谢相关论文作者提供的理论支持

## 联系方式

问题或建议请通过 GitHub Issues 联系。
