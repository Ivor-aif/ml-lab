"""频率特性实验配置文件

本文件定义了实验的所有配置参数，包括数据生成、模型结构、训练参数等。
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class ExperimentConfig:
    """实验配置类"""
    
    def __init__(self):
        # 数据生成参数
        self.data_params = {
            # 频率函数参数 f(x) = a0 + a1*sin(x+b1) + a2*sin(2x+b2) + ...
            'coefficients': {
                'a0': 2.0,      # 常数项
                'a1': 1.5,      # 1倍频幅度
                'b1': 0.0,      # 1倍频相位
                'a2': 0.8,      # 2倍频幅度
                'b2': np.pi/4,  # 2倍频相位
                'a3': 0.4,      # 3倍频幅度
                'b3': np.pi/2,  # 3倍频相位
            },
            
            # 采样参数
            'num_samples': 1000,        # 训练样本数量
            'num_test_samples': 200,    # 测试样本数量
            'x_range': (-2*np.pi, 2*np.pi),  # 采样范围
            'noise_level': 0.05,        # 噪声标准差
            'random_seed': 42,          # 随机种子
        }
        
        # 神经网络模型参数
        self.model_params = {
            'input_dim': 1,             # 输入维度
            'hidden_dim': 64,           # 隐藏层维度
            'output_dim': 1,            # 输出维度
            'activation': 'relu',       # 激活函数
            'use_bias': True,           # 是否使用偏置
            'dropout_rate': 0.1,        # Dropout率
        }
        
        # 训练参数
        self.training_params = {
            'learning_rate': 0.001,     # 学习率
            'batch_size': 32,           # 批次大小
            'num_epochs': 500,          # 训练轮数
            'validation_split': 0.2,    # 验证集比例
            'early_stopping_patience': 50,  # 早停耐心值
            'lr_scheduler': {
                'type': 'step',         # 学习率调度器类型
                'step_size': 100,       # 步长
                'gamma': 0.8,           # 衰减因子
            },
            'optimizer': 'adam',        # 优化器
            'loss_function': 'mse',     # 损失函数
        }
        
        # 可视化参数
        self.visualization_params = {
            'figure_size': (12, 8),     # 图像大小
            'dpi': 300,                 # 分辨率
            'save_format': 'png',       # 保存格式
            'plot_resolution': 1000,    # 绘图分辨率
            'colors': {
                'original': '#1f77b4',   # 原函数颜色
                'samples': '#ff7f0e',    # 采样点颜色
                'prediction': '#2ca02c', # 预测结果颜色
                'error': '#d62728',      # 误差颜色
            }
        }
        
        # 第二步实验参数 - 频域参数预测
        self.freq2_params = {
            'input_size': 200,  # 输入数据点数量 * 2 (x, y)
            'hidden_dims': [512, 256, 128],  # 隐藏层维度
            'num_freq_components': 3,  # 频率成分数量
            'learning_rate': 0.0005,  # 学习率
            'num_epochs': 500,  # 训练轮数
            'batch_size': 32,  # 批次大小
            'dropout_rate': 0.15,  # Dropout率
            'weight_decay': 1e-4,  # 权重衰减
            'param_loss_weight': 0.2,  # 参数损失权重
            'early_stopping_patience': 30,  # 早停耐心值
            'validation_split': 0.2,  # 验证集比例
            'num_data_points': 100,  # 每个样本的数据点数量
            'data_noise_level': 0.02,  # 数据噪声水平
            'x_range': (-2*np.pi, 2*np.pi),  # x值范围
            'regularization': {
                'l1_weight': 1e-5,  # L1正则化权重
                'l2_weight': 1e-4,  # L2正则化权重
                'parameter_bounds': {  # 参数边界约束
                    'a0': (-5.0, 5.0),
                    'a_max': 3.0,  # 幅度参数最大值
                    'b_range': (-np.pi, np.pi)  # 相位参数范围
                }
            }
        }
        
        # 文件路径
        self.paths = {
            'data_dir': 'data/',
            'models_dir': 'models/',
            'results_dir': 'results/',
            'plots_dir': 'results/plots/',
            'logs_dir': 'results/logs/',
            'notebooks_dir': 'notebooks/',
        }
    
    def get_frequency_components(self) -> List[Tuple[float, float, int]]:
        """获取频率成分列表
        
        Returns:
            List[Tuple[float, float, int]]: [(amplitude, phase, frequency), ...]
        """
        components = []
        coeffs = self.data_params['coefficients']
        
        # 提取频率成分
        freq = 1
        while f'a{freq}' in coeffs and f'b{freq}' in coeffs:
            amplitude = coeffs[f'a{freq}']
            phase = coeffs[f'b{freq}']
            components.append((amplitude, phase, freq))
            freq += 1
        
        return components
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置参数
        
        Args:
            updates: 要更新的参数字典
        """
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'data_params': self.data_params,
            'model_params': self.model_params,
            'training_params': self.training_params,
            'visualization_params': self.visualization_params,
            'paths': self.paths,
        }


# 默认配置实例
default_config = ExperimentConfig()