"""
模型评估和可视化模块
用于评估模型性能，观察拟合效果和震荡现象
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from model import FullyConnectedNetwork
from data_generator import DataGenerator
from trainer import TrainingHistory


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: FullyConnectedNetwork, device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        模型预测
        
        Args:
            x: 输入数据
            
        Returns:
            预测结果
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            predictions = self.model(x_tensor)
            return predictions.cpu().numpy()
    
    def compute_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算均方误差"""
        return np.mean((y_true - y_pred) ** 2)
    
    def compute_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算平均绝对误差"""
        return np.mean(np.abs(y_true - y_pred))
    
    def compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def analyze_oscillation(self, x: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        分析预测结果的震荡程度
        
        Args:
            x: 输入数据（需要排序）
            y_pred: 预测结果
            
        Returns:
            震荡分析结果
        """
        # 确保数据按x排序
        sorted_indices = np.argsort(x.flatten())
        x_sorted = x[sorted_indices]
        y_sorted = y_pred[sorted_indices]
        
        # 计算一阶导数（差分近似）
        dx = np.diff(x_sorted.flatten())
        dy = np.diff(y_sorted.flatten())
        first_derivative = dy / (dx + 1e-8)
        
        # 计算二阶导数
        d2y = np.diff(first_derivative)
        second_derivative = d2y / (dx[:-1] + 1e-8)
        
        # 震荡指标
        oscillation_metrics = {
            'max_first_derivative': np.max(np.abs(first_derivative)),
            'mean_abs_first_derivative': np.mean(np.abs(first_derivative)),
            'max_second_derivative': np.max(np.abs(second_derivative)),
            'mean_abs_second_derivative': np.mean(np.abs(second_derivative)),
            'derivative_variance': np.var(first_derivative),
            'sign_changes': np.sum(np.diff(np.sign(first_derivative)) != 0),
            'smoothness_score': 1.0 / (1.0 + np.mean(np.abs(second_derivative)))
        }
        
        return oscillation_metrics
    
    def evaluate_on_data(self, 
                        x_test: np.ndarray, 
                        y_true: np.ndarray,
                        x_train: np.ndarray = None,
                        y_train: np.ndarray = None) -> Dict:
        """
        在测试数据上评估模型
        
        Args:
            x_test: 测试输入
            y_true: 测试真实值
            x_train: 训练输入（可选）
            y_train: 训练真实值（可选）
            
        Returns:
            评估结果字典
        """
        # 测试集预测
        y_pred = self.predict(x_test)
        
        # 基本指标
        test_mse = self.compute_mse(y_true, y_pred)
        test_mae = self.compute_mae(y_true, y_pred)
        test_r2 = self.compute_r2(y_true, y_pred)
        
        # 震荡分析
        oscillation_metrics = self.analyze_oscillation(x_test, y_pred)
        
        results = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'oscillation_metrics': oscillation_metrics,
            'model_params': self.model.get_num_parameters()
        }
        
        # 如果提供了训练数据，也计算训练集指标
        if x_train is not None and y_train is not None:
            y_train_pred = self.predict(x_train)
            train_mse = self.compute_mse(y_train, y_train_pred)
            train_mae = self.compute_mae(y_train, y_train_pred)
            train_r2 = self.compute_r2(y_train, y_train_pred)
            
            results.update({
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'overfitting_ratio': test_mse / train_mse if train_mse > 0 else float('inf')
            })
        
        return results


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        初始化可视化工具
        
        Args:
            figsize: 图形大小
            style: 绘图风格
        """
        self.figsize = figsize
        plt.style.use('default')  # 使用默认风格
        sns.set_palette("husl")
    
    def plot_training_history(self, 
                            histories: Dict[str, TrainingHistory],
                            save_path: str = None):
        """
        绘制训练历史
        
        Args:
            histories: 训练历史字典 {model_name: history}
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        ax1 = axes[0, 0]
        for name, history in histories.items():
            ax1.plot(history.epochs, history.train_losses, label=name, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss vs Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 梯度范数
        ax2 = axes[0, 1]
        for name, history in histories.items():
            if history.gradient_norms:
                ax2.plot(history.epochs, history.gradient_norms, label=name, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm vs Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 参数范数
        ax3 = axes[1, 0]
        for name, history in histories.items():
            if history.parameter_norms:
                ax3.plot(history.epochs, history.parameter_norms, label=name, linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Parameter Norm')
        ax3.set_title('Parameter Norm vs Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率
        ax4 = axes[1, 1]
        for name, history in histories.items():
            if history.learning_rates:
                ax4.plot(history.epochs, history.learning_rates, label=name, linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate vs Epoch')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_function_fitting(self,
                            data_generator: DataGenerator,
                            models_and_names: List[Tuple[FullyConnectedNetwork, str]],
                            x_train: np.ndarray,
                            y_train: np.ndarray,
                            x_test: np.ndarray,
                            y_true: np.ndarray,
                            save_path: str = None):
        """
        绘制函数拟合结果
        
        Args:
            data_generator: 数据生成器
            models_and_names: 模型和名称列表
            x_train: 训练输入
            y_train: 训练输出
            x_test: 测试输入
            y_true: 真实函数值
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 绘制真实函数和训练数据
        for i, (model, name) in enumerate(models_and_names[:4]):
            ax = axes[i]
            
            # 训练数据点
            ax.scatter(x_train.flatten(), y_train.flatten(), 
                      alpha=0.6, color='blue', s=30, label='Training Data')
            
            # 真实函数
            ax.plot(x_test.flatten(), y_true.flatten(), 
                   'r-', linewidth=3, label='True Function', alpha=0.8)
            
            # 模型预测
            evaluator = ModelEvaluator(model)
            y_pred = evaluator.predict(x_test)
            ax.plot(x_test.flatten(), y_pred.flatten(), 
                   'g--', linewidth=2, label=f'{name} Prediction', alpha=0.8)
            
            # 计算指标
            results = evaluator.evaluate_on_data(x_test, y_true, x_train, y_train)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{name}\n'
                        f'Test MSE: {results["test_mse"]:.4f}, '
                        f'R²: {results["test_r2"]:.4f}\n'
                        f'Params: {results["model_params"]}, '
                        f'Smoothness: {results["oscillation_metrics"]["smoothness_score"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_oscillation_analysis(self,
                                models_and_names: List[Tuple[FullyConnectedNetwork, str]],
                                x_test: np.ndarray,
                                save_path: str = None):
        """
        绘制震荡分析图
        
        Args:
            models_and_names: 模型和名称列表
            x_test: 测试输入
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (model, name) in enumerate(models_and_names[:4]):
            ax = axes[i]
            
            evaluator = ModelEvaluator(model)
            y_pred = evaluator.predict(x_test)
            
            # 排序数据
            sorted_indices = np.argsort(x_test.flatten())
            x_sorted = x_test[sorted_indices]
            y_sorted = y_pred[sorted_indices]
            
            # 计算导数
            dx = np.diff(x_sorted.flatten())
            dy = np.diff(y_sorted.flatten())
            first_derivative = dy / (dx + 1e-8)
            
            # 绘制预测函数和导数
            ax2 = ax.twinx()
            
            ax.plot(x_sorted.flatten(), y_sorted.flatten(), 'b-', linewidth=2, label='Prediction')
            ax2.plot(x_sorted[:-1].flatten(), first_derivative, 'r--', linewidth=1, alpha=0.7, label='1st Derivative')
            
            # 分析震荡
            oscillation_metrics = evaluator.analyze_oscillation(x_test, y_pred)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y', color='b')
            ax2.set_ylabel('dy/dx', color='r')
            ax.set_title(f'{name}\n'
                        f'Max |dy/dx|: {oscillation_metrics["max_first_derivative"]:.2f}, '
                        f'Sign Changes: {oscillation_metrics["sign_changes"]}')
            
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_generalization_analysis(self,
                                   results: Dict[str, Dict],
                                   save_path: str = None):
        """
        绘制泛化能力分析图
        
        Args:
            results: 评估结果字典 {model_name: results}
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = list(results.keys())
        
        # 提取指标
        test_mses = [results[name]['test_mse'] for name in model_names]
        train_mses = [results[name].get('train_mse', 0) for name in model_names]
        r2_scores = [results[name]['test_r2'] for name in model_names]
        param_counts = [results[name]['model_params'] for name in model_names]
        smoothness_scores = [results[name]['oscillation_metrics']['smoothness_score'] for name in model_names]
        overfitting_ratios = [results[name].get('overfitting_ratio', 1) for name in model_names]
        
        # 1. 测试MSE vs 参数数量
        axes[0, 0].scatter(param_counts, test_mses, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[0, 0].annotate(name, (param_counts[i], test_mses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 0].set_xlabel('Number of Parameters')
        axes[0, 0].set_ylabel('Test MSE')
        axes[0, 0].set_title('Test MSE vs Model Complexity')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² vs 参数数量
        axes[0, 1].scatter(param_counts, r2_scores, s=100, alpha=0.7, color='green')
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (param_counts[i], r2_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Number of Parameters')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² Score vs Model Complexity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 平滑度 vs 参数数量
        axes[0, 2].scatter(param_counts, smoothness_scores, s=100, alpha=0.7, color='orange')
        for i, name in enumerate(model_names):
            axes[0, 2].annotate(name, (param_counts[i], smoothness_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 2].set_xlabel('Number of Parameters')
        axes[0, 2].set_ylabel('Smoothness Score')
        axes[0, 2].set_title('Smoothness vs Model Complexity')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 训练 vs 测试 MSE
        axes[1, 0].scatter(train_mses, test_mses, s=100, alpha=0.7, color='red')
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (train_mses[i], test_mses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        # 添加y=x线
        min_mse = min(min(train_mses), min(test_mses))
        max_mse = max(max(train_mses), max(test_mses))
        axes[1, 0].plot([min_mse, max_mse], [min_mse, max_mse], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('Train MSE')
        axes[1, 0].set_ylabel('Test MSE')
        axes[1, 0].set_title('Train vs Test MSE')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 过拟合比率
        axes[1, 1].bar(range(len(model_names)), overfitting_ratios, alpha=0.7)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Overfitting Ratio (Test MSE / Train MSE)')
        axes[1, 1].set_title('Overfitting Analysis')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 综合性能雷达图（简化版）
        axes[1, 2].remove()  # 移除最后一个子图，用于放置总结信息
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # 测试代码
    from data_generator import create_experiment_data
    from model import ModelFactory, create_overparameterized_configs
    from trainer import Trainer, create_data_loaders
    
    print("Testing evaluator and visualizer...")
    
    # 创建实验数据
    data_gen, x_train, y_train, x_test, y_test = create_experiment_data(
        degree=3, n_samples=30, noise_std=0.1
    )
    
    # 创建并训练一个简单模型
    config = {'hidden_dims': [64, 32], 'activation': 'relu', 'dropout_rate': 0.0, 'use_batch_norm': False}
    model = ModelFactory.create_model(config)
    
    # 快速训练
    train_loader, _ = create_data_loaders(x_train, y_train, batch_size=16)
    trainer = Trainer(model)
    history = trainer.train(train_loader, num_epochs=100, print_every=50)
    
    # 评估模型
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate_on_data(x_test, y_test, x_train, y_train)
    
    print("Evaluation results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    
    # 可视化
    visualizer = Visualizer()
    visualizer.plot_function_fitting(
        data_gen, [(model, 'Test Model')], 
        x_train, y_train, x_test, y_test
    )
    
    print("Evaluation and visualization completed successfully!")