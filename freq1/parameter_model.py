#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频域参数预测神经网络模型

这个模块实现了第二步实验的核心：从数据点直接预测频域参数 {a0, a1, b1, a2, b2, ...}
网络输入是一组数据点 (x, y)，输出是频域参数，然后用这些参数重构函数进行训练。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class ParameterPredictionNet(nn.Module):
    """
    频域参数预测神经网络
    
    输入: 一组数据点 (x, y) 的展平向量
    输出: 频域参数 {a0, a1, b1, a2, b2, ...}
    """
    
    def __init__(self, config):
        super(ParameterPredictionNet, self).__init__()
        self.config = config
        
        # 网络参数
        self.input_size = config.freq2_params['input_size']  # 输入数据点数量 * 2 (x, y)
        self.hidden_dims = config.freq2_params['hidden_dims']  # 隐藏层维度列表
        self.num_freq_components = config.freq2_params['num_freq_components']  # 频率成分数量
        self.output_size = 1 + 2 * self.num_freq_components  # a0 + (a1,b1) + (a2,b2) + ...
        
        # 构建网络层
        layers = []
        prev_dim = self.input_size
        
        # 隐藏层
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.freq2_params.get('dropout_rate', 0.1))
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据点，形状为 (batch_size, input_size)
        
        Returns:
            预测的频域参数，形状为 (batch_size, output_size)
        """
        return self.network(x)
    
    def predict_parameters(self, data_points):
        """
        预测频域参数
        
        Args:
            data_points: 数据点，形状为 (batch_size, num_points, 2) 或 (num_points, 2)
        
        Returns:
            预测的参数字典
        """
        self.eval()
        with torch.no_grad():
            # 处理输入形状
            if data_points.dim() == 2:
                data_points = data_points.unsqueeze(0)
            
            # 展平数据点
            batch_size = data_points.shape[0]
            flattened_input = data_points.view(batch_size, -1)
            
            # 预测参数
            raw_params = self.forward(flattened_input)
            
            # 转换为参数字典
            param_dicts = []
            for i in range(batch_size):
                params = self._raw_to_param_dict(raw_params[i])
                param_dicts.append(params)
            
            return param_dicts[0] if batch_size == 1 else param_dicts
    
    def _raw_to_param_dict(self, raw_params):
        """
        将原始输出转换为参数字典
        
        Args:
            raw_params: 网络原始输出，形状为 (output_size,)
        
        Returns:
            参数字典 {'a0': ..., 'a1': ..., 'b1': ..., ...}
        """
        param_dict = {}
        
        # 常数项
        param_dict['a0'] = raw_params[0].item()
        
        # 频率成分
        for i in range(1, self.num_freq_components + 1):
            a_idx = 1 + 2 * (i - 1)
            b_idx = 1 + 2 * (i - 1) + 1
            param_dict[f'a{i}'] = raw_params[a_idx].item()
            param_dict[f'b{i}'] = raw_params[b_idx].item()
        
        return param_dict
    
    def reconstruct_function(self, params, x_values):
        """
        使用预测的参数重构函数
        
        Args:
            params: 参数字典或参数张量
            x_values: x值数组
        
        Returns:
            重构的函数值
        """
        if isinstance(params, dict):
            # 参数字典形式
            result = torch.full_like(x_values, params['a0'])
            for i in range(1, self.num_freq_components + 1):
                if f'a{i}' in params and f'b{i}' in params:
                    result += params[f'a{i}'] * torch.sin(i * x_values + params[f'b{i}'])
        else:
            # 原始参数张量形式
            result = torch.full_like(x_values, params[0])
            for i in range(1, self.num_freq_components + 1):
                a_idx = 1 + 2 * (i - 1)
                b_idx = 1 + 2 * (i - 1) + 1
                if a_idx < len(params) and b_idx < len(params):
                    result += params[a_idx] * torch.sin(i * x_values + params[b_idx])
        
        return result

class ParameterTrainer:
    """
    频域参数预测模型训练器
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.freq2_params['learning_rate'],
            weight_decay=config.freq2_params.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=20, factor=0.5, verbose=True
        )
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'param_errors': [],
            'reconstruction_errors': []
        }
    
    def reconstruction_loss(self, predicted_params, data_points, target_values):
        """
        重构损失函数
        
        Args:
            predicted_params: 预测的参数，形状为 (batch_size, output_size)
            data_points: 输入数据点的x值，形状为 (batch_size, num_points)
            target_values: 目标y值，形状为 (batch_size, num_points)
        
        Returns:
            重构损失
        """
        batch_size = predicted_params.shape[0]
        total_loss = 0
        
        for i in range(batch_size):
            # 使用预测参数重构函数
            reconstructed = self.model.reconstruct_function(
                predicted_params[i], data_points[i]
            )
            
            # 计算重构误差
            loss = nn.MSELoss()(reconstructed, target_values[i])
            total_loss += loss
        
        return total_loss / batch_size
    
    def compute_detailed_loss(self, predicted_params, true_params, x_values, y_values):
        """
        计算详细的复合损失函数
        
        Args:
            predicted_params: 预测的参数 (batch_size, num_params)
            true_params: 真实的参数 (batch_size, num_params)
            x_values: 输入x值 (batch_size, num_points)
            y_values: 目标y值 (batch_size, num_points)
            
        Returns:
            损失字典
        """
        # 1. 参数损失 (MSE)
        param_loss = nn.MSELoss()(predicted_params, true_params)
        
        # 2. 重构损失
        recon_loss = self.reconstruction_loss(predicted_params, x_values, y_values)
        
        # 3. 一致性损失 (预测参数重构 vs 真实参数重构)
        batch_size = predicted_params.shape[0]
        consistency_loss = 0
        
        for i in range(batch_size):
            y_recon_pred = self.model.reconstruct_function(predicted_params[i], x_values[i])
            y_recon_true = self.model.reconstruct_function(true_params[i], x_values[i])
            consistency_loss += nn.MSELoss()(y_recon_pred, y_recon_true)
        
        consistency_loss = consistency_loss / batch_size
        
        # 4. 正则化损失
        reg_loss = self._compute_regularization_loss(predicted_params)
        
        # 5. 参数边界约束损失
        boundary_loss = self._compute_boundary_loss(predicted_params)
        
        # 总损失加权组合
        param_weight = self.config.freq2_params.get('param_loss_weight', 0.3)
        
        total_loss = (
            param_weight * param_loss +
            0.4 * recon_loss +
            0.2 * consistency_loss +
            0.05 * reg_loss +
            0.05 * boundary_loss
        )
        
        return {
            'total_loss': total_loss,
            'param_loss': param_loss,
            'recon_loss': recon_loss,
            'consistency_loss': consistency_loss,
            'reg_loss': reg_loss,
            'boundary_loss': boundary_loss
        }
    
    def _compute_regularization_loss(self, params):
        """
        计算正则化损失
        
        Args:
            params: 预测参数 (batch_size, num_params)
            
        Returns:
            正则化损失
        """
        # L1正则化 (稀疏性)
        l1_loss = torch.mean(torch.abs(params))
        
        # L2正则化 (平滑性)
        l2_loss = torch.mean(params ** 2)
        
        l1_weight = self.config.freq2_params.get('l1_weight', 0.001)
        l2_weight = self.config.freq2_params.get('l2_weight', 0.001)
        
        return l1_weight * l1_loss + l2_weight * l2_loss
    
    def _compute_boundary_loss(self, params):
        """
        计算参数边界约束损失
        
        Args:
            params: 预测参数 (batch_size, num_params)
            
        Returns:
            边界约束损失
        """
        # a0边界约束 (常数项)
        a0 = params[:, 0]
        a0_bound = self.config.freq2_params.get('a0_bound', 10.0)
        a0_loss = torch.mean(
            torch.relu(torch.abs(a0) - a0_bound)
        )
        
        # 幅度参数边界约束 (a1, a2, ...)
        num_components = self.config.freq2_params['num_freq_components']
        a_max = self.config.freq2_params.get('a_max', 5.0)
        
        a_loss = 0
        for i in range(1, num_components + 1):
            a_idx = 2 * i - 1  # a1在索引1, a2在索引3, ...
            if a_idx < params.shape[1]:
                a_i = params[:, a_idx]
                a_loss += torch.mean(
                    torch.relu(torch.abs(a_i) - a_max)
                )
        
        # 相位参数边界约束 (b1, b2, ...)
        b_range = self.config.freq2_params.get('b_range', [-np.pi, np.pi])
        b_loss = 0
        for i in range(1, num_components + 1):
            b_idx = 2 * i  # b1在索引2, b2在索引4, ...
            if b_idx < params.shape[1]:
                b_i = params[:, b_idx]
                b_loss += torch.mean(
                    torch.relu(b_i - b_range[1]) + torch.relu(b_range[0] - b_i)
                )
        
        return a0_loss + a_loss + b_loss
    
    def parameter_loss(self, predicted_params, true_params):
        """
        参数损失函数（如果有真实参数的话）
        
        Args:
            predicted_params: 预测参数
            true_params: 真实参数
        
        Returns:
            参数损失
        """
        return nn.MSELoss()(predicted_params, true_params)
    
    def train_epoch(self, train_loader, val_loader=None):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        
        Returns:
            训练损失和验证损失
        """
        self.model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_data in train_loader:
            # 解包数据
            if len(batch_data) == 3:
                data_points, x_values, y_values = batch_data
                true_params = None
            else:
                data_points, x_values, y_values, true_params = batch_data
            
            # 移动到设备
            data_points = data_points.to(self.device)
            x_values = x_values.to(self.device)
            y_values = y_values.to(self.device)
            if true_params is not None:
                true_params = true_params.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predicted_params = self.model(data_points.view(data_points.shape[0], -1))
            
            # 计算损失
            if true_params is not None:
                # 使用详细损失函数
                loss_dict = self.compute_detailed_loss(predicted_params, true_params, x_values, y_values)
                total_loss = loss_dict['total_loss']
            else:
                # 只有重构损失
                recon_loss = self.reconstruction_loss(predicted_params, x_values, y_values)
                total_loss = recon_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            train_loss += total_loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # 验证
        val_loss = 0
        if val_loader is not None:
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
        
        return avg_train_loss, val_loss
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            验证损失
        """
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # 解包数据
                if len(batch_data) == 3:
                    data_points, x_values, y_values = batch_data
                    true_params = None
                else:
                    data_points, x_values, y_values, true_params = batch_data
                
                # 移动到设备
                data_points = data_points.to(self.device)
                x_values = x_values.to(self.device)
                y_values = y_values.to(self.device)
                if true_params is not None:
                    true_params = true_params.to(self.device)
                
                # 前向传播
                predicted_params = self.model(data_points.view(data_points.shape[0], -1))
                
                # 计算损失
                if true_params is not None:
                    # 使用详细损失函数
                    loss_dict = self.compute_detailed_loss(predicted_params, true_params, x_values, y_values)
                    total_loss = loss_dict['total_loss']
                else:
                    # 只有重构损失
                    recon_loss = self.reconstruction_loss(predicted_params, x_values, y_values)
                    total_loss = recon_loss
                
                val_loss += total_loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, train_loader, val_loader=None, num_epochs=None, verbose=True):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            verbose: 是否显示训练过程
        
        Returns:
            训练历史
        """
        if num_epochs is None:
            num_epochs = self.config.freq2_params['num_epochs']
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.freq2_params.get('early_stopping_patience', 50)
        
        for epoch in range(num_epochs):
            train_loss, val_loss = self.train_epoch(train_loader, val_loader)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            if val_loader is not None:
                self.training_history['val_loss'].append(val_loss)
            
            # 早停检查
            if val_loader is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 显示进度
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        return self.training_history
    
    def evaluate(self, test_loader):
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            评估结果字典
        """
        self.model.eval()
        total_recon_error = 0
        total_param_error = 0
        num_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    data_points, x_values, y_values = batch_data
                    true_params = None
                else:
                    data_points, x_values, y_values, true_params = batch_data
                
                data_points = data_points.to(self.device)
                x_values = x_values.to(self.device)
                y_values = y_values.to(self.device)
                
                # 预测参数
                predicted_params = self.model(data_points.view(data_points.shape[0], -1))
                
                # 重构误差
                recon_error = self.reconstruction_loss(predicted_params, x_values, y_values)
                total_recon_error += recon_error.item() * data_points.shape[0]
                
                # 参数误差（如果有真实参数）
                if true_params is not None:
                    true_params = true_params.to(self.device)
                    param_error = self.parameter_loss(predicted_params, true_params)
                    total_param_error += param_error.item() * data_points.shape[0]
                
                num_samples += data_points.shape[0]
                
                # 收集预测结果
                all_predictions.extend(predicted_params.cpu().numpy())
                if true_params is not None:
                    all_targets.extend(true_params.cpu().numpy())
        
        results = {
            'reconstruction_error': total_recon_error / num_samples,
            'num_samples': num_samples
        }
        
        if total_param_error > 0:
            results['parameter_error'] = total_param_error / num_samples
        
        return results
    
    def save_checkpoint(self, filepath):
        """
        保存模型检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """
        加载模型检查点
        
        Args:
            filepath: 检查点路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']

def load_parameter_model(filepath, config):
    """
    加载训练好的参数预测模型
    
    Args:
        filepath: 模型文件路径
        config: 配置对象
    
    Returns:
        加载的模型
    """
    model = ParameterPredictionNet(config)
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    # 简单测试
    from config import ExperimentConfig
    
    # 创建配置（需要添加freq2_params）
    config = ExperimentConfig()
    
    # 添加第二步实验参数
    config.freq2_params = {
        'input_size': 200,  # 100个数据点 * 2 (x, y)
        'hidden_dims': [256, 128, 64],
        'num_freq_components': 3,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'dropout_rate': 0.1,
        'weight_decay': 1e-5,
        'param_loss_weight': 0.1,
        'early_stopping_patience': 20
    }
    
    # 创建模型
    model = ParameterPredictionNet(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"输入维度: {model.input_size}")
    print(f"输出维度: {model.output_size}")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, model.input_size)
    output = model(test_input)
    print(f"输出形状: {output.shape}")
    
    # 测试参数预测
    test_data_points = torch.randn(100, 2)  # 100个数据点
    predicted_params = model.predict_parameters(test_data_points)
    print(f"预测参数: {predicted_params}")