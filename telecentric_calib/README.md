# telecentric_calib

远心相机标定包。计算按推导顺序拆开，每个文件只做一步。

```text
telecentric_calib/
  params.py            # CalibParamHolder 与标定板世界坐标
  detect.py            # 圆点检测，含阈值重试
  halcon_detector/     # FindCirclesGridPattern / FindHalconCalibBoard
  single/
    homography.py      # 单应矩阵与内参
    extrinsics.py      # r31、r32 与旋转矩阵
    optimize.py        # 带畸变的整体优化
    calibrate.py       # 按顺序调用上面四步，加上检测和保存
  stereo.py            # 双目标定与重建误差
  io.py                # 图像读写和 JSON
  config.py            # YAML 配置
  api/                 # 给目录直接跑检测或完整标定的薄封装
```

```python
from telecentric_calib.params import CalibParamHolder
from telecentric_calib.single import TelecentricCalibrator

param_holder = CalibParamHolder()
param_holder.set_board(
    dot_array_shape=(7, 7),
    dot_diameter=0.1875,
    dot_spacing=0.375,
)
calibrator = TelecentricCalibrator(param_holder, "image_dir")
calibrator.single_camera_calibration()
calibrator.save_camera_params_to_json("output.json")
```

命令行示例在仓库根目录的 `examples/`。
