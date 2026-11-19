English | [中文](README.md)

# Telecentric Camera Calibration

## Overview

Python library for calibrating telecentric cameras, both monocular and stereo. The planar-target method follows Zhong Chen (2014) and Hai Li (2024).

## Features

- Monocular calibration: intrinsics, extrinsics, and radial distortion
- Stereo calibration and 3D reconstruction for a telecentric pair
- YAML configuration for paths and target geometry
- Verbosity controlled by `verbose`
- Image IO that accepts non-ASCII paths
- Library code lives in `src/telecentric_calib`

## Installation

### Requirements

- Python >= 3.7
- Windows, Linux, or macOS

### Steps

1. Clone this repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The examples add `src` to `sys.path`, so they run after the dependencies are installed. To import `telecentric_calib` from another directory:

```bash
pip install -e .
```

3. Create a config file:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your image directories and target parameters.

## Quick start

Set the image directories, dot-array shape, diameter, spacing, and output names in `config.yaml`.

Monocular calibration:

```bash
python examples/example_single_camera.py
```

Stereo calibration:

```bash
python examples/example_dual_camera.py
```

## Usage

### Monocular

1. Capture at least 4 views of the target in different poses. Ten or more is a practical choice.
2. Set the image directory and target parameters in `config.yaml`.
3. Run `examples/example_single_camera.py`.
4. Read the JSON file written to the output directory.

The linear intrinsic solve needs at least 4 homographies. Images are taken in directory listing order and are not sorted.

### Stereo

1. Calibrate the left and right cameras separately.
2. Capture one synchronized pair of the same target.
3. Set the stereo directory and file names in `config.yaml`.
4. Run `examples/example_dual_camera.py`.
5. The outputs are the projection matrices, the fundamental matrix, and the reconstruction errors.

### Configuration

Main fields of `config.yaml`:

```yaml
paths:
  left_camera_images: "image/left"
  right_camera_images: "image/right"
  binocular_images: "image/binocular"
  output_dir: "output"

calibration_board:
  dot_array_shape: [7, 7]               # rows, columns
  dot_diameter: 0.1875                  # millimetres
  dot_spacing: 0.375                    # centre spacing, millimetres

binocular:
  left_image_name: "0_left.bmp"
  right_image_name: "0_right.bmp"

output:
  verbose: true
  left_camera_output: "leftCKmatrix.json"
  right_camera_output: "rightCKmatrix.json"
```

## Layout

```
telecentric-camera-calibration/
├── README.md                      # Chinese
├── README.en.md                   # this file
├── LICENSE                        # BSD-2-Clause
├── requirements.txt
├── pyproject.toml
├── config.example.yaml
├── docs/
│   └── 数学推导.md                # derivation, Chinese
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
    ├── detect/
    ├── mono/
    └── stereo.py
```

## Pattern detection

Circle detection lives in `telecentric_calib.detect`. It is a Python port of [find_calib_pattern](https://github.com/Neucrede/find_calib_pattern).

- Upstream: https://github.com/Neucrede/find_calib_pattern
- License: BSD-2-Clause
- Public functions: `FindCirclesGridPattern`, `FindHalconCalibBoard`

Calibration tries a plain circle grid first, then a HALCON calibration board. The public entry points are `calibrate_mono` and `calibrate_stereo`.

## Mathematics

The derivation is in [docs/数学推导.md](docs/数学推导.md) (Chinese). The intrinsic recovery, the linear system, and the rotation-sign step match the code. The note also records the stereo reconstruction equations.

The implementation follows:

1. Chen, Z., et al. (2014). Telecentric stereo micro-vision system: Calibration method and experiments.
2. Li, H., et al. (2024). Flexible Calibration of the Telecentric Vision Systems Using Only Planar Calibration Target.

Monocular calibration solves the homography, the intrinsics, and the full rotation, then refines every parameter together with radial distortion.

## Citation

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

## References

1. Chen, Z., et al. (2014). Telecentric stereo micro-vision system: Calibration method and experiments.

2. Li, H., Liao, Z., Cai, W., et al. (2024). Flexible Calibration of the Telecentric Vision Systems Using Only Planar Calibration Target. IEEE Transactions on Instrumentation and Measurement, 73, 1-10.

3. Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision. Cambridge university press.

## Test images

The capture sets are large and are not in this repository. Prepare your own images.

- Monocular: at least 4 poses, preferably 10 or more
- Stereo: a monocular set for each camera, plus one synchronized pair
- Target: a plain circle grid or a HALCON circle board, preferably 7×7 or larger

## License

BSD-2-Clause. See [LICENSE](LICENSE).

## Acknowledgements

- [Neucrede/find_calib_pattern](https://github.com/Neucrede/find_calib_pattern) for the board detector, now in `telecentric_calib.detect`
- The authors of the papers cited above

## Contact

Open a GitHub issue for questions or suggestions.
