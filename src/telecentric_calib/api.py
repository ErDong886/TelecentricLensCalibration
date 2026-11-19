"""对外的标定入口。"""
from .io import save_camera_params
from .mono import MonoCalibration, calibrate, make_logger
from .stereo import DualCameraCalibrator


def calibrate_mono(image_dir, param_holder, verbose=True) -> MonoCalibration:
    """标定一台远心相机。"""
    return calibrate(image_dir, param_holder, verbose)


def save_mono_params(calibration: MonoCalibration, output_path, image_index=0):
    """保存一台相机的内参和外参。"""
    return save_camera_params(
        calibration.final_params,
        output_path,
        image_index,
        make_logger(calibration.verbose),
    )


def calibrate_stereo(
    left: MonoCalibration,
    right: MonoCalibration,
    binocular_image_dir,
    left_image_name="0_left.bmp",
    right_image_name="0_right.bmp",
    verbose=True,
):
    """用两台已标定相机和一对同步图像做双目标定。"""
    worker = DualCameraCalibrator(
        left,
        right,
        binocular_image_dir,
        left_image_name=left_image_name,
        right_image_name=right_image_name,
        verbose=verbose,
    )
    return worker.dual_camera_calibration()


def save_fundamental_matrix(points_left, points_right, output_path, verbose=True):
    """由同名点计算并保存基础矩阵。"""
    worker = DualCameraCalibrator(None, None, "", verbose=verbose)
    return worker.save_fundamental_matrix_json(points_left, points_right, output_path)


def save_reconstruction_errors(
    points_theory,
    points_reconstructed_distorted,
    points_reconstructed_undistorted,
    output_path,
    verbose=True,
):
    """保存三维重建误差。"""
    worker = DualCameraCalibrator(None, None, "", verbose=verbose)
    return worker.save_reconstruction_errors_json(
        points_theory,
        points_reconstructed_distorted,
        points_reconstructed_undistorted,
        output_path,
    )
