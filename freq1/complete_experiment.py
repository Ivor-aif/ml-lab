#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频率特性实验 - 完整实验代码

该脚本整合了两个实验步骤：
1. 第一步：神经网络拟合频率函数 (x,y) 映射
2. 第二步：神经网络直接预测频域参数 {a0, a1, b1, a2, b2, ...}

增强的可视化功能：
- 训练过程中输出与数据点的演化对比
- 输出与原函数的差别展示
- 参数预测精度分析
- 函数重构质量评估

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_generator import FrequencyDataGenerator
from model import FrequencyNet, FrequencyTrainer
from parameter_model import ParameterPredictionNet, ParameterTrainer
from utils import setup_logging, save_results, create_directory

class CompleteFrequencyExperiment:
    """
    完整的频率特性实验类
    整合两个实验步骤并提供增强的可视化功能
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging('complete_experiment')
        
        # 创建必要的目录
        create_directory(self.config.paths['results'])
        create_directory(self.config.paths['models'])
        create_directory(self.config.paths['plots'])
        
        # 初始化组件
        self.data_generator = FrequencyDataGenerator(config)
        
        # 实验结果存储
        self.results = {
            'step1_results': {},  # 第一步实验结果
            'step2_results': {},  # 第二步实验结果
            'comparison_metrics': {},  # 对比指标
            'experiment_config': {
                'step1_config': config.model_params,
                'step2_config': config.freq2_params
            }
        }
        
        # 可视化历史记录
        self.visualization_history = {
            'step1_training_snapshots': [],
            'step2_training_snapshots': [],
            'evolution_data': []
        }
    
    def generate_experiment_data(self, num_samples: int = 1000) -> Tuple[Dict, Dict]:
        """
        生成实验数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            step1_data: 第一步实验数据
            step2_data: 第二步实验数据
        """
        self.logger.info(f"生成 {num_samples} 个实验样本...")
        
        # 生成参数和函数数据
        all_params = []
        step1_X = []
        step1_y = []
        step2_X = []
        step2_y = []
        
        num_components = self.config.freq2_params['num_freq_components']
        num_points = self.config.freq2_params['num_data_points']
        x_range = self.config.freq2_params['x_range']
        noise_level = self.config.freq2_params['data_noise_level']
        
        # 生成x坐标
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        
        for i in tqdm(range(num_samples), desc="生成数据"):
            # 生成随机参数
            params = self._generate_random_parameters(num_components)
            all_params.append(params)
            
            # 生成真实函数值
            y_true = self.data_generator.generate_frequency_function(x_coords, params)
            
            # 添加噪声
            y_noisy = y_true + np.random.normal(0, noise_level, len(y_true))
            
            # 第一步数据：(x, y) 映射
            for j in range(len(x_coords)):
                step1_X.append([x_coords[j]])
                step1_y.append([y_noisy[j]])
            
            # 第二步数据：数据点 -> 参数
            xy_data = np.concatenate([x_coords, y_noisy])
            param_array = self._params_dict_to_array(params, num_components)
            
            step2_X.append(xy_data)
            step2_y.append(param_array)
        
        step1_data = {
            'X': np.array(step1_X),
            'y': np.array(step1_y),
            'x_coords': x_coords,
            'original_functions': all_params
        }
        
        step2_data = {
            'X': np.array(step2_X),
            'y': np.array(step2_y),
            'x_coords': x_coords,
            'original_params': all_params
        }
        
        self.logger.info(f"数据生成完成: Step1 X.shape={step1_data['X'].shape}, Step2 X.shape={step2_data['X'].shape}")
        return step1_data, step2_data
    
    def run_step1_experiment(self, step1_data: Dict) -> Dict:
        """
        运行第一步实验：函数拟合
        
        Args:
            step1_data: 第一步实验数据
            
        Returns:
            第一步实验结果
        """
        self.logger.info("开始第一步实验：函数拟合...")
        
        # 划分数据集
        X, y = step1_data['X'], step1_data['y']
        split_idx = int(0.8 * len(X))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 创建模型
        model = FrequencyNet(
            input_size=self.config.model_params['input_size'],
            hidden_size=self.config.model_params['hidden_size'],
            output_size=self.config.model_params['output_size']
        )
        
        # 创建训练器
        trainer = FrequencyTrainer(model, self.config)
        
        # 训练模型（带可视化回调）
        history = trainer.train(
            X_train, y_train, X_test, y_test,
            visualization_callback=self._step1_visualization_callback
        )
        
        # 评估模型
        test_loss = trainer.evaluate(X_test, y_test)
        
        # 生成预测
        predictions = trainer.predict(X_test)
        
        step1_results = {
            'model': model,
            'trainer': trainer,
            'history': history,
            'test_loss': test_loss,
            'predictions': predictions,
            'test_data': {'X': X_test, 'y': y_test}
        }
        
        self.results['step1_results'] = step1_results
        self.logger.info(f"第一步实验完成，测试损失: {test_loss:.6f}")
        return step1_results
    
    def run_step2_experiment(self, step2_data: Dict) -> Dict:
        """
        运行第二步实验：参数预测
        
        Args:
            step2_data: 第二步实验数据
            
        Returns:
            第二步实验结果
        """
        self.logger.info("开始第二步实验：参数预测...")
        
        # 划分数据集
        X, y = step2_data['X'], step2_data['y']
        split_idx = int(0.8 * len(X))
        val_split_idx = int(0.6 * len(X))
        
        X_train = X[val_split_idx:split_idx]
        y_train = y[val_split_idx:split_idx]
        X_val = X[:val_split_idx]
        y_val = y[:val_split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # 创建模型
        input_size = self.config.freq2_params['input_size']
        hidden_dims = self.config.freq2_params['hidden_dims']
        num_components = self.config.freq2_params['num_freq_components']
        output_size = 1 + 2 * num_components
        dropout_rate = self.config.freq2_params['dropout_rate']
        
        model = ParameterPredictionNet(
            input_size=input_size,
            hidden_dims=hidden_dims,
            output_size=output_size,
            dropout_rate=dropout_rate
        )
        
        # 创建训练器
        trainer = ParameterTrainer(model, self.config)
        
        # 训练模型（带可视化回调）
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            visualization_callback=self._step2_visualization_callback
        )
        
        # 评估模型
        metrics = trainer.evaluate(X_test, y_test)
        
        # 生成预测样例
        predictions = []
        for i in range(min(10, len(X_test))):
            pred_params = model.predict_parameters(X_test[i:i+1])[0]
            true_params = step2_data['original_params'][split_idx + i]
            
            predictions.append({
                'predicted_params': self._array_to_params_dict(pred_params, num_components),
                'true_params': true_params,
                'input_data': X_test[i]
            })
        
        step2_results = {
            'model': model,
            'trainer': trainer,
            'history': history,
            'metrics': metrics,
            'predictions': predictions,
            'test_data': {'X': X_test, 'y': y_test, 'original_params': step2_data['original_params'][split_idx:]}
        }
        
        self.results['step2_results'] = step2_results
        self.logger.info(f"第二步实验完成，测试指标: {metrics}")
        return step2_results
    
    def _step1_visualization_callback(self, epoch: int, model: Any, train_loss: float, val_loss: float):
        """
        第一步实验的可视化回调函数
        """
        # 每10个epoch记录一次训练快照
        if epoch % 10 == 0:
            snapshot = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now()
            }
            self.visualization_history['step1_training_snapshots'].append(snapshot)
    
    def _step2_visualization_callback(self, epoch: int, model: Any, train_loss: float, val_loss: float):
        """
        第二步实验的可视化回调函数
        """
        # 每20个epoch记录一次训练快照
        if epoch % 20 == 0:
            snapshot = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now()
            }
            self.visualization_history['step2_training_snapshots'].append(snapshot)
    
    def create_enhanced_visualizations(self, step1_data: Dict, step2_data: Dict):
        """
        创建增强的可视化图表
        
        Args:
            step1_data: 第一步实验数据
            step2_data: 第二步实验数据
        """
        self.logger.info("生成增强可视化图表...")
        
        # 设置绘图样式
        plt.style.use(self.config.visualization_params['style'])
        sns.set_palette(self.config.visualization_params['color_palette'])
        
        # 1. 训练演化对比图
        self._plot_training_evolution()
        
        # 2. 函数拟合质量对比
        self._plot_function_fitting_comparison(step1_data)
        
        # 3. 参数预测精度分析
        self._plot_parameter_prediction_analysis(step2_data)
        
        # 4. 原函数vs预测函数对比
        self._plot_original_vs_predicted_functions(step2_data)
        
        # 5. 综合性能对比
        self._plot_comprehensive_performance_comparison()
        
        self.logger.info("可视化图表生成完成")
    
    def _plot_training_evolution(self):
        """
        绘制训练演化过程
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('训练演化过程对比', fontsize=16, fontweight='bold')
        
        # 第一步训练历史
        if 'step1_results' in self.results and 'history' in self.results['step1_results']:
            history1 = self.results['step1_results']['history']
            axes[0, 0].plot(history1['train_loss'], label='训练损失', alpha=0.8, linewidth=2)
            axes[0, 0].plot(history1['val_loss'], label='验证损失', alpha=0.8, linewidth=2)
            axes[0, 0].set_title('第一步：函数拟合训练历史', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
        
        # 第二步训练历史
        if 'step2_results' in self.results and 'history' in self.results['step2_results']:
            history2 = self.results['step2_results']['history']
            axes[0, 1].plot(history2['train_loss'], label='训练损失', alpha=0.8, linewidth=2)
            if 'val_loss' in history2:
                axes[0, 1].plot(history2['val_loss'], label='验证损失', alpha=0.8, linewidth=2)
            axes[0, 1].set_title('第二步：参数预测训练历史', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        
        # 训练快照对比
        if self.visualization_history['step1_training_snapshots']:
            snapshots = self.visualization_history['step1_training_snapshots']
            epochs = [s['epoch'] for s in snapshots]
            train_losses = [s['train_loss'] for s in snapshots]
            val_losses = [s['val_loss'] for s in snapshots]
            
            axes[1, 0].plot(epochs, train_losses, 'o-', label='训练损失快照', alpha=0.8, markersize=6)
            axes[1, 0].plot(epochs, val_losses, 's-', label='验证损失快照', alpha=0.8, markersize=6)
            axes[1, 0].set_title('第一步训练快照', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if self.visualization_history['step2_training_snapshots']:
            snapshots = self.visualization_history['step2_training_snapshots']
            epochs = [s['epoch'] for s in snapshots]
            train_losses = [s['train_loss'] for s in snapshots]
            val_losses = [s['val_loss'] for s in snapshots]
            
            axes[1, 1].plot(epochs, train_losses, 'o-', label='训练损失快照', alpha=0.8, markersize=6)
            axes[1, 1].plot(epochs, val_losses, 's-', label='验证损失快照', alpha=0.8, markersize=6)
            axes[1, 1].set_title('第二步训练快照', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.paths['plots'], 'training_evolution.png'),
                   dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _plot_function_fitting_comparison(self, step1_data: Dict):
        """
        绘制函数拟合质量对比
        """
        if 'step1_results' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('第一步：函数拟合质量对比（输出 vs 数据点 vs 原函数）', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        step1_results = self.results['step1_results']
        model = step1_results['model']
        x_coords = step1_data['x_coords']
        original_functions = step1_data['original_functions']
        
        # 选择6个样例进行对比
        num_examples = min(6, len(original_functions))
        selected_indices = np.random.choice(len(original_functions), num_examples, replace=False)
        
        for i, func_idx in enumerate(selected_indices):
            # 获取原函数参数
            original_params = original_functions[func_idx]
            
            # 生成原函数
            y_original = self.data_generator.generate_frequency_function(x_coords, original_params)
            
            # 生成带噪声的数据点
            noise_level = self.config.freq2_params['data_noise_level']
            y_noisy = y_original + np.random.normal(0, noise_level, len(y_original))
            
            # 模型预测
            X_pred = np.array([[x] for x in x_coords])
            y_pred = model.predict(X_pred).flatten()
            
            # 绘制对比
            axes[i].plot(x_coords, y_original, 'b-', label='原函数', linewidth=2.5, alpha=0.9)
            axes[i].plot(x_coords, y_pred, 'r--', label='模型输出', linewidth=2.5, alpha=0.9)
            axes[i].scatter(x_coords, y_noisy, c='green', s=25, alpha=0.7, label='数据点', zorder=5)
            
            # 计算误差
            mse_vs_original = np.mean((y_pred - y_original) ** 2)
            mse_vs_data = np.mean((y_pred - y_noisy) ** 2)
            
            axes[i].set_title(f'样例 {i+1}\nMSE(vs原函数): {mse_vs_original:.4f}\nMSE(vs数据): {mse_vs_data:.4f}', 
                            fontsize=10, fontweight='bold')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.paths['plots'], 'function_fitting_comparison.png'),
                   dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _plot_parameter_prediction_analysis(self, step2_data: Dict):
        """
        绘制参数预测精度分析
        """
        if 'step2_results' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('第二步：参数预测精度分析', fontsize=16, fontweight='bold')
        
        predictions = self.results['step2_results']['predictions']
        
        # 提取参数对比数据
        param_names = ['a0', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3']
        true_values = {name: [] for name in param_names}
        pred_values = {name: [] for name in param_names}
        
        for pred in predictions:
            true_params = pred['true_params']
            pred_params = pred['predicted_params']
            
            for name in param_names:
                if name in true_params and name in pred_params:
                    true_values[name].append(true_params[name])
                    pred_values[name].append(pred_params[name])
        
        # 参数散点图对比
        for i, param_name in enumerate(['a0', 'a1', 'b1', 'a2']):
            if len(true_values[param_name]) > 0:
                ax = axes[i//2, i%2]
                
                true_vals = np.array(true_values[param_name])
                pred_vals = np.array(pred_values[param_name])
                
                # 散点图
                ax.scatter(true_vals, pred_vals, alpha=0.7, s=50)
                
                # 理想线
                min_val = min(np.min(true_vals), np.min(pred_vals))
                max_val = max(np.max(true_vals), np.max(pred_vals))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # 计算相关系数
                correlation = np.corrcoef(true_vals, pred_vals)[0, 1]
                mse = np.mean((true_vals - pred_vals) ** 2)
                
                ax.set_title(f'参数 {param_name}\n相关系数: {correlation:.3f}, MSE: {mse:.4f}', 
                           fontweight='bold')
                ax.set_xlabel(f'真实 {param_name}')
                ax.set_ylabel(f'预测 {param_name}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.paths['plots'], 'parameter_prediction_analysis.png'),
                   dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _plot_original_vs_predicted_functions(self, step2_data: Dict):
        """
        绘制原函数vs预测函数对比
        """
        if 'step2_results' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('第二步：原函数 vs 预测函数对比', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        predictions = self.results['step2_results']['predictions']
        x_coords = step2_data['x_coords']
        num_points = len(x_coords)
        
        for i, pred in enumerate(predictions[:6]):
            true_params = pred['true_params']
            pred_params = pred['predicted_params']
            input_data = pred['input_data']
            
            # 提取输入数据点
            x_input = input_data[:num_points]
            y_input = input_data[num_points:]
            
            # 生成函数
            y_true = self.data_generator.generate_frequency_function(x_coords, true_params)
            y_pred = self.data_generator.generate_frequency_function(x_coords, pred_params)
            
            # 绘制对比
            axes[i].plot(x_coords, y_true, 'b-', label='原函数', linewidth=2.5, alpha=0.9)
            axes[i].plot(x_coords, y_pred, 'r--', label='预测函数', linewidth=2.5, alpha=0.9)
            axes[i].scatter(x_input, y_input, c='green', s=25, alpha=0.7, label='输入数据点', zorder=5)
            
            # 计算误差指标
            mse_func = np.mean((y_true - y_pred) ** 2)
            mse_data = np.mean((y_input - y_pred) ** 2)
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            axes[i].set_title(f'样例 {i+1}\nMSE(函数): {mse_func:.4f}\n相关性: {correlation:.3f}', 
                            fontsize=10, fontweight='bold')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.paths['plots'], 'original_vs_predicted_functions.png'),
                   dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _plot_comprehensive_performance_comparison(self):
        """
        绘制综合性能对比
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('综合性能对比分析', fontsize=16, fontweight='bold')
        
        # 损失对比
        if 'step1_results' in self.results and 'step2_results' in self.results:
            step1_final_loss = self.results['step1_results']['test_loss']
            step2_metrics = self.results['step2_results']['metrics']
            
            methods = ['第一步\n(函数拟合)', '第二步\n(参数预测)']
            losses = [step1_final_loss, step2_metrics.get('test_loss', 0)]
            
            axes[0, 0].bar(methods, losses, alpha=0.8, color=['skyblue', 'lightcoral'])
            axes[0, 0].set_title('最终测试损失对比', fontweight='bold')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 训练时间对比（如果有记录）
        if hasattr(self, 'training_times'):
            axes[0, 1].bar(methods, self.training_times, alpha=0.8, color=['lightgreen', 'orange'])
            axes[0, 1].set_title('训练时间对比', fontweight='bold')
            axes[0, 1].set_ylabel('时间 (秒)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 参数预测精度分布
        if 'step2_results' in self.results:
            predictions = self.results['step2_results']['predictions']
            param_errors = []
            
            for pred in predictions:
                true_params = pred['true_params']
                pred_params = pred['predicted_params']
                
                for param_name in ['a0', 'a1', 'a2']:
                    if param_name in true_params and param_name in pred_params:
                        error = abs(true_params[param_name] - pred_params[param_name])
                        param_errors.append(error)
            
            if param_errors:
                axes[1, 0].hist(param_errors, bins=20, alpha=0.8, color='lightblue', edgecolor='black')
                axes[1, 0].set_title('参数预测误差分布', fontweight='bold')
                axes[1, 0].set_xlabel('绝对误差')
                axes[1, 0].set_ylabel('频次')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 模型复杂度对比
        if 'step1_results' in self.results and 'step2_results' in self.results:
            step1_params = sum(p.numel() for p in self.results['step1_results']['model'].parameters())
            step2_params = sum(p.numel() for p in self.results['step2_results']['model'].parameters())
            
            axes[1, 1].bar(['第一步模型', '第二步模型'], [step1_params, step2_params], 
                          alpha=0.8, color=['gold', 'mediumpurple'])
            axes[1, 1].set_title('模型参数数量对比', fontweight='bold')
            axes[1, 1].set_ylabel('参数数量')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.paths['plots'], 'comprehensive_performance.png'),
                   dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _generate_random_parameters(self, num_components: int) -> Dict:
        """
        生成随机频域参数
        """
        bounds = self.config.freq2_params['regularization']['parameter_bounds']
        
        params = {
            'a0': np.random.uniform(bounds['a0'][0], bounds['a0'][1])
        }
        
        a_max = bounds['a_max']
        b_range = bounds['b_range']
        
        for i in range(1, num_components + 1):
            params[f'a{i}'] = np.random.uniform(-a_max, a_max)
            params[f'b{i}'] = np.random.uniform(b_range[0], b_range[1])
        
        return params
    
    def _params_dict_to_array(self, params: Dict, num_components: int) -> np.ndarray:
        """
        将参数字典转换为数组
        """
        param_array = [params['a0']]
        
        for i in range(1, num_components + 1):
            param_array.extend([params[f'a{i}'], params[f'b{i}']])
        
        return np.array(param_array)
    
    def _array_to_params_dict(self, param_array: np.ndarray, num_components: int) -> Dict:
        """
        将参数数组转换为字典
        """
        params = {'a0': param_array[0]}
        
        idx = 1
        for i in range(1, num_components + 1):
            params[f'a{i}'] = param_array[idx]
            params[f'b{i}'] = param_array[idx + 1]
            idx += 2
        
        return params
    
    def save_complete_results(self):
        """
        保存完整实验结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.config.paths['results'], 
                                   f'complete_experiment_results_{timestamp}.json')
        
        # 转换为可序列化格式
        serializable_results = self._make_json_serializable(self.results)
        
        save_results(serializable_results, results_file)
        self.logger.info(f"完整实验结果已保存到: {results_file}")
        
        # 保存模型
        if 'step1_results' in self.results:
            model_file = os.path.join(self.config.paths['models'], 
                                     f'step1_model_{timestamp}.pth')
            self.results['step1_results']['trainer'].save_checkpoint(model_file)
        
        if 'step2_results' in self.results:
            model_file = os.path.join(self.config.paths['models'], 
                                     f'step2_model_{timestamp}.pth')
            self.results['step2_results']['trainer'].save_checkpoint(model_file)
    
    def _make_json_serializable(self, obj):
        """
        将对象转换为JSON可序列化格式
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items() 
                   if key not in ['model', 'trainer']}  # 排除不可序列化的对象
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # 转换为字符串表示
        else:
            return obj
    
    def run_complete_experiment(self, num_samples: int = 1000):
        """
        运行完整的两步实验
        
        Args:
            num_samples: 样本数量
        """
        self.logger.info("开始完整的频率特性实验...")
        
        try:
            # 1. 生成实验数据
            step1_data, step2_data = self.generate_experiment_data(num_samples)
            
            # 2. 运行第一步实验
            step1_results = self.run_step1_experiment(step1_data)
            
            # 3. 运行第二步实验
            step2_results = self.run_step2_experiment(step2_data)
            
            # 4. 创建增强可视化
            self.create_enhanced_visualizations(step1_data, step2_data)
            
            # 5. 保存结果
            self.save_complete_results()
            
            self.logger.info("完整实验成功完成!")
            
            # 打印总结
            self._print_experiment_summary()
            
        except Exception as e:
            self.logger.error(f"实验过程中发生错误: {str(e)}")
            raise
    
    def _print_experiment_summary(self):
        """
        打印实验总结
        """
        print("\n" + "=" * 80)
        print("频率特性实验完整总结")
        print("=" * 80)
        
        if 'step1_results' in self.results:
            step1_loss = self.results['step1_results']['test_loss']
            print(f"第一步实验（函数拟合）:")
            print(f"  - 最终测试损失: {step1_loss:.6f}")
            print(f"  - 模型类型: 两层全连接网络")
        
        if 'step2_results' in self.results:
            step2_metrics = self.results['step2_results']['metrics']
            print(f"\n第二步实验（参数预测）:")
            print(f"  - 测试指标: {step2_metrics}")
            print(f"  - 模型类型: 多层感知机")
        
        print(f"\n可视化图表已保存到: {self.config.paths['plots']}")
        print(f"实验结果已保存到: {self.config.paths['results']}")
        print(f"模型文件已保存到: {self.config.paths['models']}")
        print("\n实验完成! 🎉")

def main():
    """
    主函数
    """
    print("=" * 80)
    print("频率特性实验 - 完整实验代码")
    print("整合第一步（函数拟合）和第二步（参数预测）")
    print("=" * 80)
    
    # 初始化配置
    config = Config()
    
    # 创建实验实例
    experiment = CompleteFrequencyExperiment(config)
    
    # 运行完整实验
    experiment.run_complete_experiment(num_samples=800)
    
    print("\n完整实验已完成! 请查看生成的图表和结果文件。")

if __name__ == "__main__":
    main()