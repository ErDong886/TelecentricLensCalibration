# 远心相机标定项目（重构版）

## 项目简介

这是一个用于远心相机标定的Python项目，支持单目和双目标定。项目已经重构，采用模块化设计，便于维护和扩展。

## 项目结构

```
refactored/
├── src/                    # 核心源代码
│   ├── __init__.py
│   ├── params.py          # 参数管理模块
│   ├── detector.py        # 检测器模块（使用github_halcondetector）
│   ├── calibrator.py      # 标定器核心模块
│   └── utils.py           # 工具函数模块
├── api/                    # API接口
│   ├── __init__.py
│   └── calibration_api.py # 高级API接口
├── examples/               # 示例代码
│   └── example_single_camera.py
├── tests/                  # 测试代码
├── docs/                   # 文档
└── README.md              # 本文件
```

## 主要模块说明

### 1. params.py
参数管理模块，包含 `CalibParamHolder` 类，用于存储和管理相机标定参数（内参、外参、标定板参数）。

### 2. detector.py
检测器模块，使用 `github_halcondetector` 进行标定板检测。包含 `PatternBoard` 类和 `detect_pattern` 函数。

### 3. calibrator.py
核心标定器模块，包含：
- `TelecentricCalibrator`: 单相机标定器
- `DualCameraCalibrator`: 双目标定器

### 4. utils.py
工具函数模块，包含：
- 图像IO函数（支持中文路径）
- JSON处理函数
- 可视化函数

### 5. api/calibration_api.py
高级API接口，提供简化的标定接口。

## 使用方法

### 单相机标定

```python
from src.params import CalibParamHolder
from src.calibrator import TelecentricCalibrator

# 创建参数容器
param_holder = CalibParamHolder()
param_holder.set_board(
    dot_array_shape=(7, 7),
    dot_diameter=0.1875,
    dot_spacing=0.375
)

# 创建标定器
calibrator = TelecentricCalibrator(param_holder, "image_dir")

# 执行标定
final_params = calibrator.single_camera_calibration()

# 保存结果
calibrator.save_camera_params_to_json("output.json")
```

详细示例请参考 `examples/example_single_camera.py`。

## 依赖说明

项目依赖以下模块：
- `github_halcondetector`: 标定板检测器（位于项目根目录）
- `opencv-python`: 图像处理
- `numpy`: 数值计算
- `scipy`: 科学计算（用于优化）

## 注意事项

1. **检测器**: 项目使用 `github_halcondetector` 进行标定板检测，不再使用 `corner_detector`。
2. **路径处理**: 图像读取和保存使用 `utils.py` 中的函数，支持中文路径。
3. **编码问题**: 使用OpenCV和matplotlib时，注意中文显示问题（已配置SimHei字体）。

## 重构说明

相比原项目，重构后的主要改进：
1. **模块化设计**: 代码按功能模块组织，便于维护
2. **清晰的层次结构**: src/api/examples/tests分离
3. **统一的接口**: 使用统一的API接口
4. **改进的检测器**: 只使用github_halcondetector，不再使用corner_detector

## 开发状态

项目正在重构中，部分功能可能需要进一步完善。

## 许可证

（根据原项目许可证）

