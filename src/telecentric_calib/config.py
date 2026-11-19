"""
配置管理模块
用于加载和管理标定配置
"""
import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        'paths': {
            'left_camera_images': 'image/20251106leftcalib',
            'right_camera_images': 'image/20251106rightcalib',
            'binocular_images': 'image/20251106binocular',
            'output_dir': 'output'
        },
        'calibration_board': {
            'dot_array_shape': [7, 7],
            'dot_diameter': 0.1875,
            'dot_spacing': 0.375
        },
        'binocular': {
            'left_image_name': '0_left.bmp',
            'right_image_name': '0_right.bmp'
        },
        'output': {
            'verbose': True,
            'left_camera_output': 'leftCKmatrix.json',
            'right_camera_output': 'rightCKmatrix.json',
            'fundamental_matrix_output': 'fundamentalMatrix.json',
            'reconstruction_errors_output': 'reconstruction_errors.json'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path is not None and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            # 尝试从当前目录加载config.yaml
            default_paths = ['config.yaml', '../config.yaml', '../../config.yaml']
            for path in default_paths:
                if os.path.exists(path):
                    self.load_from_file(path)
                    break
    
    def load_from_file(self, config_path: str):
        """
        从YAML文件加载配置
        
        参数:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_config(self.config, user_config)
        except Exception as e:
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
            print("将使用默认配置")
    
    def _merge_config(self, base: Dict, update: Dict):
        """
        递归合并配置字典
        
        参数:
            base: 基础配置
            update: 更新配置
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        参数:
            key_path: 配置路径，如 'paths.left_camera_images'
            default: 默认值
        
        返回:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值（支持点号分隔的路径）
        
        参数:
            key_path: 配置路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_path(self, path_key: str, create: bool = False) -> str:
        """
        获取路径配置（自动处理相对路径）
        
        参数:
            path_key: 路径配置键
            create: 如果路径不存在是否创建
        
        返回:
            绝对路径
        """
        path = self.get(f'paths.{path_key}')
        if path is None:
            return None
        
        # 如果不是绝对路径，转换为绝对路径
        if not os.path.isabs(path):
            # 相对于项目根目录
            path = os.path.abspath(path)
        
        # 如果需要创建且不存在
        if create and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        return path


# 全局配置实例
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    获取全局配置实例
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置实例
    """
    global _global_config
    
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    
    return _global_config


def load_config(config_path: str) -> Config:
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置实例
    """
    return Config(config_path)

