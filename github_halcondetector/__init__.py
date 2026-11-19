"""兼容旧的导入路径。检测实现在 telecentric_calib.halcon_detector。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from telecentric_calib.halcon_detector import FindCirclesGridPattern, FindHalconCalibBoard

__version__ = "1.0.0"
__all__ = ["FindCirclesGridPattern", "FindHalconCalibBoard"]
