"""可视化工具

本模块提供了实验结果的可视化功能，包括：
- 原函数与拟合结果对比
- 训练过程可视化
- 频率成分分析
- 误差分析
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, Any, Optional, Tuple, List

from config import ExperimentConfig
from data_generator import FrequencyDataGenerator
from model import TwoLayerNet

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class ExperimentVisualizer:
    """实验可视化器"""
    
    def __init__(self, config: ExperimentConfig):
        """初始化可视化器
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.vis_params = config.visualization_params
        self.colors = self.vis_params['colors']
        
        # 设置图像参数
        plt.rcParams['figure.figsize'] = self.vis_params['figure_size']
        plt.rcParams['figure.dpi'] = self.vis_params['dpi']
    
    def plot_function_fitting(self, 
                             generator: FrequencyDataGenerator,
                             model: TwoLayerNet,
                             data: Dict[str, np.ndarray],
                             save_path: Optional[str] = None) -> None:
        """绘制函数拟合结果
        
        Args:
            generator: 数据生成器
            model: 训练好的模型
            data: 数据字典
            save_path: 保存路径
        """
        # 生成高分辨率数据用于绘制
        x_high_res, y_true_high_res = generator.generate_high_resolution_data(
            self.vis_params['plot_resolution']
        )
        
        # 模型预测
        y_pred_high_res = model.predict(x_high_res)
        y_pred_train = model.predict(data['x_train'])
        y_pred_test = model.predict(data['x_test'])
        
        # 创建子图
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 整体拟合效果
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x_high_res.flatten(), y_true_high_res.flatten(), 
                label='原函数', color=self.colors['original'], linewidth=2.5)
        ax1.plot(x_high_res.flatten(), y_pred_high_res.flatten(), 
                label='神经网络拟合', color=self.colors['prediction'], 
                linewidth=2, linestyle='--', alpha=0.8)
        ax1.scatter(data['x_train'].flatten(), data['y_train'].flatten(), 
                   label='训练样本', color=self.colors['samples'], 
                   alpha=0.4, s=15)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('频率函数拟合结果', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 训练数据拟合
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(data['y_train'].flatten(), y_pred_train.flatten(), 
                   color=self.colors['samples'], alpha=0.6, s=20)
        
        # 添加理想拟合线
        min_val = min(data['y_train'].min(), y_pred_train.min())
        max_val = max(data['y_train'].max(), y_pred_train.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'k--', alpha=0.8, linewidth=1)
        
        ax2.set_xlabel('真实值')
        ax2.set_ylabel('预测值')
        ax2.set_title('训练数据拟合散点图')
        ax2.grid(True, alpha=0.3)
        
        # 计算R²
        ss_res = np.sum((data['y_train'] - y_pred_train) ** 2)
        ss_tot = np.sum((data['y_train'] - np.mean(data['y_train'])) ** 2)
        r2_train = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        ax2.text(0.05, 0.95, f'R² = {r2_train:.4f}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. 测试数据拟合
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(data['y_test'].flatten(), y_pred_test.flatten(), 
                   color=self.colors['prediction'], alpha=0.6, s=20)
        
        # 添加理想拟合线
        min_val = min(data['y_test'].min(), y_pred_test.min())
        max_val = max(data['y_test'].max(), y_pred_test.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 
                'k--', alpha=0.8, linewidth=1)
        
        ax3.set_xlabel('真实值')
        ax3.set_ylabel('预测值')
        ax3.set_title('测试数据拟合散点图')
        ax3.grid(True, alpha=0.3)
        
        # 计算R²
        ss_res = np.sum((data['y_test'] - y_pred_test) ** 2)
        ss_tot = np.sum((data['y_test'] - np.mean(data['y_test'])) ** 2)
        r2_test = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        ax3.text(0.05, 0.95, f'R² = {r2_test:.4f}', 
                transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('神经网络频率函数拟合分析', fontsize=16, fontweight='bold')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"拟合结果图已保存到: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, 
                             training_history: Dict[str, List],
                             save_path: Optional[str] = None) -> None:
        """绘制训练历史
        
        Args:
            training_history: 训练历史数据
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 1. 损失曲线
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        axes[0].plot(epochs, training_history['train_loss'], 
                    label='训练损失', color=self.colors['samples'], linewidth=2)
        
        if 'val_loss' in training_history and training_history['val_loss']:
            axes[0].plot(epochs, training_history['val_loss'], 
                        label='验证损失', color=self.colors['prediction'], linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('训练损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # 2. 学习率曲线
        if 'learning_rates' in training_history and training_history['learning_rates']:
            axes[1].plot(epochs, training_history['learning_rates'], 
                        color=self.colors['error'], linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('学习率变化')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        else:
            axes[1].text(0.5, 0.5, '无学习率数据', 
                        transform=axes[1].transAxes, 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('学习率变化')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def plot_frequency_analysis(self, 
                               generator: FrequencyDataGenerator,
                               save_path: Optional[str] = None) -> None:
        """绘制频率成分分析
        
        Args:
            generator: 数据生成器
            save_path: 保存路径
        """
        analysis = generator.analyze_frequency_components()
        components = analysis['frequency_components']
        
        if not components:
            print("没有频率成分可以分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 频率成分幅度
        frequencies = [comp['frequency'] for comp in components]
        amplitudes = [comp['amplitude'] for comp in components]
        
        axes[0, 0].bar(frequencies, amplitudes, color=self.colors['samples'], alpha=0.7)
        axes[0, 0].set_xlabel('频率')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].set_title('各频率成分幅度')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 相位分布
        phases = [comp['phase_degrees'] for comp in components]
        axes[0, 1].bar(frequencies, phases, color=self.colors['prediction'], alpha=0.7)
        axes[0, 1].set_xlabel('频率')
        axes[0, 1].set_ylabel('相位 (度)')
        axes[0, 1].set_title('各频率成分相位')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 贡献比例饼图
        contributions = [comp['contribution_ratio'] for comp in components]
        labels = [f'频率 {comp["frequency"]}' for comp in components]
        
        axes[1, 0].pie(contributions, labels=labels, autopct='%1.1f%%', 
                      colors=plt.cm.Set3(np.linspace(0, 1, len(components))))
        axes[1, 0].set_title('各频率成分贡献比例')
        
        # 4. 单独频率成分可视化
        x_range = generator.config.data_params['x_range']
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        for i, comp in enumerate(components[:3]):  # 只显示前3个成分
            freq = comp['frequency']
            amp = comp['amplitude']
            phase = comp['phase']
            y = amp * np.sin(freq * x + phase)
            
            axes[1, 1].plot(x, y, label=f'频率 {freq}', linewidth=2)
        
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('幅度')
        axes[1, 1].set_title('主要频率成分波形')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"频率分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_error_analysis(self, 
                           generator: FrequencyDataGenerator,
                           model: TwoLayerNet,
                           data: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> None:
        """绘制误差分析
        
        Args:
            generator: 数据生成器
            model: 训练好的模型
            data: 数据字典
            save_path: 保存路径
        """
        # 生成高分辨率预测
        x_high_res, y_true_high_res = generator.generate_high_resolution_data(1000)
        y_pred_high_res = model.predict(x_high_res)
        
        # 计算误差
        error = y_true_high_res - y_pred_high_res
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 误差随位置变化
        axes[0, 0].plot(x_high_res.flatten(), error.flatten(), 
                       color=self.colors['error'], linewidth=1.5)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('误差 (真实值 - 预测值)')
        axes[0, 0].set_title('预测误差分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差直方图
        axes[0, 1].hist(error.flatten(), bins=50, color=self.colors['error'], 
                       alpha=0.7, density=True)
        axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('误差')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].set_title('误差分布直方图')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(error)
        std_error = np.std(error)
        axes[0, 1].text(0.05, 0.95, f'均值: {mean_error:.4f}\n标准差: {std_error:.4f}', 
                        transform=axes[0, 1].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
        
        # 3. 绝对误差
        abs_error = np.abs(error)
        axes[1, 0].plot(x_high_res.flatten(), abs_error.flatten(), 
                       color=self.colors['prediction'], linewidth=1.5)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('绝对误差')
        axes[1, 0].set_title('绝对误差分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 相对误差（避免除零）
        relative_error = np.where(np.abs(y_true_high_res) > 1e-8, 
                                 error / y_true_high_res * 100, 0)
        axes[1, 1].plot(x_high_res.flatten(), relative_error.flatten(), 
                       color=self.colors['original'], linewidth=1.5)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('相对误差 (%)')
        axes[1, 1].set_title('相对误差分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"误差分析图已保存到: {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, 
                                   generator: FrequencyDataGenerator,
                                   model: TwoLayerNet,
                                   data: Dict[str, np.ndarray],
                                   training_history: Dict[str, List],
                                   save_dir: str) -> None:
        """创建综合实验报告
        
        Args:
            generator: 数据生成器
            model: 训练好的模型
            data: 数据字典
            training_history: 训练历史
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("生成综合实验报告...")
        
        # 1. 函数拟合结果
        self.plot_function_fitting(
            generator, model, data, 
            os.path.join(save_dir, 'function_fitting.png')
        )
        
        # 2. 训练历史
        self.plot_training_history(
            training_history, 
            os.path.join(save_dir, 'training_history.png')
        )
        
        # 3. 频率分析
        self.plot_frequency_analysis(
            generator, 
            os.path.join(save_dir, 'frequency_analysis.png')
        )
        
        # 4. 误差分析
        self.plot_error_analysis(
            generator, model, data, 
            os.path.join(save_dir, 'error_analysis.png')
        )
        
        print(f"综合实验报告已保存到: {save_dir}")


def main():
    """主函数：演示可视化功能"""
    from config import ExperimentConfig
    from train import load_model
    import torch
    
    # 检查是否有训练好的模型
    model_path = 'results/freq1_baseline/model.pth'
    if not os.path.exists(model_path):
        print("未找到训练好的模型，请先运行 train.py")
        return
    
    # 加载配置和模型
    config = ExperimentConfig()
    model = load_model(model_path, config)
    
    # 生成数据
    generator = FrequencyDataGenerator(config)
    data = generator.generate_train_test_data()
    
    # 加载训练历史
    import json
    history_path = 'results/freq1_baseline/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    else:
        training_history = {'train_loss': [], 'val_loss': []}
    
    # 创建可视化器
    visualizer = ExperimentVisualizer(config)
    
    # 生成综合报告
    visualizer.create_comprehensive_report(
        generator, model, data, training_history, 
        'results/freq1_baseline/plots'
    )


if __name__ == '__main__':
    main()