"""两层神经网络模型

本模块实现了用于拟合频率函数的两层神经网络模型。
模型结构：输入层 -> 隐藏层 -> 输出层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from config import ExperimentConfig


class TwoLayerNet(nn.Module):
    """两层神经网络模型
    
    网络结构：
    - 输入层：接收1维输入 (x值)
    - 隐藏层：全连接层 + 激活函数 + Dropout
    - 输出层：全连接层，输出1维结果 (y值)
    """
    
    def __init__(self, config: ExperimentConfig):
        """初始化网络
        
        Args:
            config: 实验配置对象
        """
        super(TwoLayerNet, self).__init__()
        
        self.config = config
        model_params = config.model_params
        
        # 网络层定义
        self.input_dim = model_params['input_dim']
        self.hidden_dim = model_params['hidden_dim']
        self.output_dim = model_params['output_dim']
        
        # 第一层：输入到隐藏层
        self.fc1 = nn.Linear(
            self.input_dim, 
            self.hidden_dim, 
            bias=model_params['use_bias']
        )
        
        # 第二层：隐藏层到输出层
        self.fc2 = nn.Linear(
            self.hidden_dim, 
            self.output_dim, 
            bias=model_params['use_bias']
        )
        
        # Dropout层
        self.dropout = nn.Dropout(model_params['dropout_rate'])
        
        # 激活函数
        self.activation = self._get_activation_function(model_params['activation'])
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_activation_function(self, activation_name: str) -> nn.Module:
        """获取激活函数
        
        Args:
            activation_name: 激活函数名称
            
        Returns:
            nn.Module: 激活函数
        """
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        
        if activation_name.lower() not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
        return activation_map[activation_name.lower()]
    
    def _initialize_weights(self) -> None:
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, output_dim]
        """
        # 第一层：输入 -> 隐藏层
        h1 = self.fc1(x)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)
        
        # 第二层：隐藏层 -> 输出
        output = self.fc2(h1)
        
        return output
    
    def get_hidden_representation(self, x: torch.Tensor) -> torch.Tensor:
        """获取隐藏层表示
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 隐藏层输出
        """
        with torch.no_grad():
            h1 = self.fc1(x)
            h1 = self.activation(h1)
            return h1
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测函数
        
        Args:
            x: 输入数组 [num_samples, input_dim]
            
        Returns:
            np.ndarray: 预测结果 [num_samples, output_dim]
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            predictions = self.forward(x_tensor)
            return predictions.numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TwoLayerNet',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'activation_function': str(self.activation),
            'dropout_rate': self.config.model_params['dropout_rate'],
            'use_bias': self.config.model_params['use_bias'],
        }


class ModelTrainer:
    """模型训练器
    
    负责模型的训练、验证和评估过程。
    """
    
    def __init__(self, model: TwoLayerNet, config: ExperimentConfig):
        """初始化训练器
        
        Args:
            model: 神经网络模型
            config: 实验配置
        """
        self.model = model
        self.config = config
        self.training_params = config.training_params
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_loss_function()
        self.scheduler = self._get_scheduler()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """获取优化器
        
        Returns:
            torch.optim.Optimizer: 优化器
        """
        optimizer_name = self.training_params['optimizer'].lower()
        lr = self.training_params['learning_rate']
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _get_loss_function(self) -> nn.Module:
        """获取损失函数
        
        Returns:
            nn.Module: 损失函数
        """
        loss_name = self.training_params['loss_function'].lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """获取学习率调度器
        
        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: 学习率调度器
        """
        scheduler_config = self.training_params.get('lr_scheduler')
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_params['num_epochs']
            )
        else:
            return None
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 前向传播
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        num_epochs = self.training_params['num_epochs']
        patience = self.training_params.get('early_stopping_patience', float('inf'))
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_history['train_loss'].append(train_loss)
            
            # 验证
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.train_history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.training_params['learning_rate']
            
            self.train_history['learning_rates'].append(current_lr)
            
            # 打印进度
            if (epoch + 1) % 50 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"LR: {current_lr:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.6f}, "
                          f"LR: {current_lr:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"恢复最佳模型，验证损失: {best_val_loss:.6f}")
        
        return {
            'final_train_loss': self.train_history['train_loss'][-1],
            'best_val_loss': best_val_loss if val_loader else None,
            'total_epochs': len(self.train_history['train_loss']),
            'train_history': self.train_history
        }
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            Dict[str, float]: 评估指标
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算评估指标
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # 计算R²分数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score
        }


def create_model(config: ExperimentConfig) -> TwoLayerNet:
    """创建模型实例
    
    Args:
        config: 实验配置
        
    Returns:
        TwoLayerNet: 模型实例
    """
    model = TwoLayerNet(config)
    print("模型创建成功:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return model


if __name__ == '__main__':
    # 测试模型创建
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    model = create_model(config)
    
    # 测试前向传播
    test_input = torch.randn(10, 1)
    output = model(test_input)
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"测试输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")