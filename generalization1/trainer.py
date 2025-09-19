"""
训练器模块
实现神经网络的训练过程，记录训练误差和相关指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import time
import json
from pathlib import Path

from model import FullyConnectedNetwork
from data_generator import DataGenerator


class TrainingHistory:
    """训练历史记录类"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        self.gradient_norms = []
        self.parameter_norms = []
        
    def add_epoch(self, 
                  epoch: int,
                  train_loss: float,
                  val_loss: float = None,
                  lr: float = None,
                  grad_norm: float = None,
                  param_norm: float = None):
        """添加一个epoch的记录"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
        if param_norm is not None:
            self.parameter_norms.append(param_norm)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms,
            'parameter_norms': self.parameter_norms
        }
    
    def save(self, filepath: str):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """加载训练历史"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        history = cls()
        history.epochs = data.get('epochs', [])
        history.train_losses = data.get('train_losses', [])
        history.val_losses = data.get('val_losses', [])
        history.learning_rates = data.get('learning_rates', [])
        history.gradient_norms = data.get('gradient_norms', [])
        history.parameter_norms = data.get('parameter_norms', [])
        
        return history


class Trainer:
    """神经网络训练器"""
    
    def __init__(self, 
                 model: FullyConnectedNetwork,
                 device: str = 'cpu',
                 save_dir: str = './results'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 训练设备
            save_dir: 保存目录
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.history = TrainingHistory()
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def compute_gradient_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def compute_parameter_norm(self) -> float:
        """计算参数范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, 
                val_loader: torch.utils.data.DataLoader,
                criterion: nn.Module) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader = None,
              num_epochs: int = 1000,
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              scheduler_type: str = None,
              early_stopping_patience: int = None,
              print_every: int = 100,
              save_best: bool = True) -> TrainingHistory:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            optimizer_type: 优化器类型
            scheduler_type: 学习率调度器类型
            early_stopping_patience: 早停耐心值
            print_every: 打印间隔
            save_best: 是否保存最佳模型
            
        Returns:
            训练历史
        """
        # 设置损失函数
        criterion = nn.MSELoss()
        
        # 设置优化器
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 设置学习率调度器
        scheduler = None
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        
        # 早停相关变量
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.get_num_parameters()}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader, criterion)
            
            # 计算梯度和参数范数
            grad_norm = self.compute_gradient_norm()
            param_norm = self.compute_parameter_norm()
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history.add_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=current_lr,
                grad_norm=grad_norm,
                param_norm=param_norm
            )
            
            # 更新学习率
            if scheduler is not None:
                if scheduler_type == 'plateau' and val_loss is not None:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 保存最佳模型
            if save_best and val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            elif early_stopping_patience is not None:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % print_every == 0 or epoch == 0:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1:4d}/{num_epochs}: "
                      f"Train Loss: {train_loss:.6f}{val_str}, "
                      f"LR: {current_lr:.2e}, "
                      f"Grad Norm: {grad_norm:.4f}")
            
            # 早停检查
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # 恢复最佳模型
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return self.history
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'hidden_dims': self.model.hidden_dims,
                'activation': self.model.activation,
                'dropout_rate': self.model.dropout_rate,
                'use_batch_norm': self.model.use_batch_norm
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def create_data_loaders(x_train: np.ndarray, 
                       y_train: np.ndarray,
                       x_val: np.ndarray = None,
                       y_val: np.ndarray = None,
                       batch_size: int = 32,
                       shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    创建数据加载器
    
    Args:
        x_train: 训练输入数据
        y_train: 训练输出数据
        x_val: 验证输入数据
        y_val: 验证输出数据
        batch_size: 批大小
        shuffle: 是否打乱数据
        
    Returns:
        (train_loader, val_loader)
    """
    # 转换为张量
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    val_loader = None
    if x_val is not None and y_val is not None:
        x_val_tensor = torch.FloatTensor(x_val)
        y_val_tensor = torch.FloatTensor(y_val)
        val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试代码
    from data_generator import create_experiment_data
    from model import ModelFactory, create_overparameterized_configs
    
    print("Testing trainer...")
    
    # 创建实验数据
    data_gen, x_train, y_train, x_test, y_test = create_experiment_data(
        degree=3, n_samples=30, noise_std=0.1
    )
    
    # 创建过参数化模型配置
    configs = create_overparameterized_configs(n_samples=30)
    config = configs[0]  # 选择第一个配置
    
    # 创建模型
    model = ModelFactory.create_model(config)
    print(f"Model: {config['name']}")
    print(f"Parameters: {model.get_num_parameters()}")
    print(f"Training samples: {len(x_train)}")
    
    # 创建数据加载器
    train_loader, _ = create_data_loaders(x_train, y_train, batch_size=16)
    
    # 创建训练器
    trainer = Trainer(model, device='cpu')
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=500,
        learning_rate=0.001,
        print_every=100
    )
    
    print(f"Final training loss: {history.train_losses[-1]:.6f}")
    print("Training completed successfully!")