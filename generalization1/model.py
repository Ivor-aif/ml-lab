"""
神经网络模型模块
实现L层全连接神经网络，用于函数拟合实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class FullyConnectedNetwork(nn.Module):
    """L层全连接神经网络"""
    
    def __init__(self, 
                 input_dim: int = 1,
                 output_dim: int = 1,
                 hidden_dims: List[int] = [64, 64],
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False):
        """
        初始化全连接网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表，长度决定了网络层数
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用批归一化
        """
        super(FullyConnectedNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        # 输入层到第一个隐藏层
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 最后一层到输出
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # 输出层使用较小的初始化
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 批归一化
            if self.use_batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # 激活函数
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(x)
            
            # Dropout
            if self.dropout_rate > 0 and self.dropouts is not None:
                x = self.dropouts[i](x)
        
        # 输出层（不使用激活函数）
        x = self.output_layer(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_info(self) -> str:
        """获取网络架构信息"""
        info = f"Network Architecture:\n"
        info += f"Input dim: {self.input_dim}\n"
        
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            info += f"Hidden layer {i+1}: {prev_dim} -> {hidden_dim} ({self.activation})\n"
            prev_dim = hidden_dim
        
        info += f"Output layer: {prev_dim} -> {self.output_dim}\n"
        info += f"Total parameters: {self.get_num_parameters()}\n"
        info += f"Activation: {self.activation}\n"
        info += f"Dropout rate: {self.dropout_rate}\n"
        info += f"Batch normalization: {self.use_batch_norm}\n"
        
        return info


def create_network_configs(max_layers: int = 5, 
                          base_width: int = 64) -> List[dict]:
    """
    创建不同层数的网络配置
    
    Args:
        max_layers: 最大层数
        base_width: 基础宽度
        
    Returns:
        网络配置列表
    """
    configs = []
    
    for num_layers in range(1, max_layers + 1):
        # 配置1: 固定宽度
        config = {
            'name': f'L{num_layers}_fixed_width',
            'hidden_dims': [base_width] * num_layers,
            'activation': 'relu',
            'dropout_rate': 0.0,
            'use_batch_norm': False
        }
        configs.append(config)
        
        # 配置2: 递减宽度
        if num_layers > 1:
            dims = []
            for i in range(num_layers):
                dim = max(16, base_width // (2 ** i))
                dims.append(dim)
            
            config = {
                'name': f'L{num_layers}_decreasing_width',
                'hidden_dims': dims,
                'activation': 'relu',
                'dropout_rate': 0.0,
                'use_batch_norm': False
            }
            configs.append(config)
    
    return configs


def create_overparameterized_configs(n_samples: int) -> List[dict]:
    """
    创建过参数化的网络配置（参数数量 > 样本数量）
    
    Args:
        n_samples: 训练样本数量
        
    Returns:
        过参数化网络配置列表
    """
    configs = []
    
    # 计算不同配置下的参数数量，确保超过样本数量
    base_configs = [
        {'layers': 2, 'width': 128},  # 约 16K+ 参数
        {'layers': 3, 'width': 64},   # 约 12K+ 参数
        {'layers': 4, 'width': 32},   # 约 4K+ 参数
        {'layers': 5, 'width': 24},   # 约 3K+ 参数
        {'layers': 6, 'width': 16},   # 约 1.5K+ 参数
    ]
    
    for config in base_configs:
        # 估算参数数量
        layers = config['layers']
        width = config['width']
        
        # 粗略估算: input->hidden + hidden->hidden*(layers-1) + hidden->output
        approx_params = 1 * width + width * width * (layers - 1) + width * 1
        
        if approx_params > n_samples:
            network_config = {
                'name': f'L{layers}_W{width}_overparameterized',
                'hidden_dims': [width] * layers,
                'activation': 'relu',
                'dropout_rate': 0.0,
                'use_batch_norm': False,
                'estimated_params': approx_params
            }
            configs.append(network_config)
    
    return configs


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(config: dict, 
                    input_dim: int = 1, 
                    output_dim: int = 1) -> FullyConnectedNetwork:
        """
        根据配置创建模型
        
        Args:
            config: 模型配置字典
            input_dim: 输入维度
            output_dim: 输出维度
            
        Returns:
            创建的模型
        """
        return FullyConnectedNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config['hidden_dims'],
            activation=config.get('activation', 'relu'),
            dropout_rate=config.get('dropout_rate', 0.0),
            use_batch_norm=config.get('use_batch_norm', False)
        )
    
    @staticmethod
    def print_model_comparison(configs: List[dict], 
                              n_samples: int,
                              input_dim: int = 1,
                              output_dim: int = 1):
        """
        打印模型配置比较
        
        Args:
            configs: 配置列表
            n_samples: 样本数量
            input_dim: 输入维度
            output_dim: 输出维度
        """
        print(f"Model Comparison (Training samples: {n_samples})")
        print("=" * 80)
        print(f"{'Model Name':<25} {'Layers':<8} {'Width':<15} {'Parameters':<12} {'Overparameterized':<15}")
        print("-" * 80)
        
        for config in configs:
            model = ModelFactory.create_model(config, input_dim, output_dim)
            num_params = model.get_num_parameters()
            is_over = "Yes" if num_params > n_samples else "No"
            width_str = str(config['hidden_dims'])
            
            print(f"{config['name']:<25} {len(config['hidden_dims']):<8} {width_str:<15} {num_params:<12} {is_over:<15}")


if __name__ == "__main__":
    # 测试代码
    print("Testing neural network models...")
    
    # 创建不同配置
    configs = create_network_configs(max_layers=4, base_width=64)
    
    # 假设有30个训练样本
    n_samples = 30
    overparameterized_configs = create_overparameterized_configs(n_samples)
    
    # 打印模型比较
    all_configs = configs + overparameterized_configs
    ModelFactory.print_model_comparison(all_configs, n_samples)
    
    # 创建一个示例模型
    config = overparameterized_configs[0]
    model = ModelFactory.create_model(config)
    
    print(f"\nExample model architecture:")
    print(model.get_architecture_info())
    
    # 测试前向传播
    x = torch.randn(10, 1)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")