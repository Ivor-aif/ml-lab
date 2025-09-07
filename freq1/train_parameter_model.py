#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频率特性实验 - 第二步：频域参数拟合训练脚本

该脚本实现了从数据点直接拟合频域参数的完整训练流程：
1. 生成训练数据集
2. 训练参数预测神经网络
3. 评估模型性能
4. 可视化拟合结果

作者: AI Assistant
日期: 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_generator import FrequencyDataGenerator
from parameter_model import ParameterPredictionNet, ParameterTrainer
from utils import setup_logging, save_results, create_directory

class ParameterFittingExperiment:
    """
    频域参数拟合实验主类
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging('parameter_fitting')
        
        # 创建必要的目录
        create_directory(self.config.paths['results'])
        create_directory(self.config.paths['models'])
        create_directory(self.config.paths['plots'])
        
        # 初始化组件
        self.data_generator = FrequencyDataGenerator(config)
        self.model = None
        self.trainer = None
        
        # 实验结果存储
        self.results = {
            'training_history': {},
            'evaluation_metrics': {},
            'best_predictions': [],
            'experiment_config': config.freq2_params
        }
    
    def generate_training_dataset(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据集
        
        Args:
            num_samples: 样本数量
            
        Returns:
            X: 输入数据 (num_samples, num_data_points * 2)
            y: 目标参数 (num_samples, num_params)
        """
        self.logger.info(f"生成 {num_samples} 个训练样本...")
        
        X_data = []
        y_params = []
        
        num_components = self.config.freq2_params['num_freq_components']
        num_points = self.config.freq2_params['num_data_points']
        x_range = self.config.freq2_params['x_range']
        noise_level = self.config.freq2_params['data_noise_level']
        
        # 生成x坐标
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        for i in tqdm(range(num_samples), desc="生成数据"):
            # 随机生成频域参数
            params = self._generate_random_parameters(num_components)
            
            # 生成对应的函数值
            y_true = self.data_generator.generate_frequency_function(x, params)
            
            # 添加噪声
            y_noisy = y_true + np.random.normal(0, noise_level, len(y_true))
            
            # 组合输入数据 (x, y)
            xy_data = np.concatenate([x, y_noisy])
            
            X_data.append(xy_data)
            y_params.append(self._params_dict_to_array(params, num_components))
        
        X = np.array(X_data)
        y = np.array(y_params)
        
        self.logger.info(f"数据集生成完成: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    
    def _generate_random_parameters(self, num_components: int) -> Dict:
        """
        生成随机频域参数
        
        Args:
            num_components: 频率成分数量
            
        Returns:
            参数字典
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
        
        Args:
            params: 参数字典
            num_components: 频率成分数量
            
        Returns:
            参数数组
        """
        param_array = [params['a0']]
        
        for i in range(1, num_components + 1):
            param_array.extend([params[f'a{i}'], params[f'b{i}']])
        
        return np.array(param_array)
    
    def _array_to_params_dict(self, param_array: np.ndarray, num_components: int) -> Dict:
        """
        将参数数组转换为字典
        
        Args:
            param_array: 参数数组
            num_components: 频率成分数量
            
        Returns:
            参数字典
        """
        params = {'a0': param_array[0]}
        
        idx = 1
        for i in range(1, num_components + 1):
            params[f'a{i}'] = param_array[idx]
            params[f'b{i}'] = param_array[idx + 1]
            idx += 2
        
        return params
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        训练参数预测模型
        
        Args:
            X_train: 训练输入数据
            y_train: 训练目标参数
            X_val: 验证输入数据
            y_val: 验证目标参数
        """
        self.logger.info("开始训练参数预测模型...")
        
        # 初始化模型
        input_size = self.config.freq2_params['input_size']
        hidden_dims = self.config.freq2_params['hidden_dims']
        num_components = self.config.freq2_params['num_freq_components']
        output_size = 1 + 2 * num_components  # a0 + (a1,b1) + (a2,b2) + ...
        dropout_rate = self.config.freq2_params['dropout_rate']
        
        self.model = ParameterPredictionNet(
            input_size=input_size,
            hidden_dims=hidden_dims,
            output_size=output_size,
            dropout_rate=dropout_rate
        )
        
        # 初始化训练器
        self.trainer = ParameterTrainer(
            model=self.model,
            config=self.config
        )
        
        # 训练模型
        history = self.trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        self.results['training_history'] = history
        self.logger.info("模型训练完成")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        评估模型性能
        
        Args:
            X_test: 测试输入数据
            y_test: 测试目标参数
            
        Returns:
            评估指标字典
        """
        self.logger.info("评估模型性能...")
        
        if self.trainer is None:
            raise ValueError("模型尚未训练")
        
        metrics = self.trainer.evaluate(X_test, y_test)
        self.results['evaluation_metrics'] = metrics
        
        # 生成一些预测样例
        num_examples = min(10, len(X_test))
        indices = np.random.choice(len(X_test), num_examples, replace=False)
        
        for idx in indices:
            x_sample = X_test[idx:idx+1]
            y_true = y_test[idx]
            y_pred = self.model.predict_parameters(x_sample)[0]
            
            # 转换为参数字典格式
            num_components = self.config.freq2_params['num_freq_components']
            true_params = self._array_to_params_dict(y_true, num_components)
            pred_params = self._array_to_params_dict(y_pred, num_components)
            
            self.results['best_predictions'].append({
                'true_params': true_params,
                'pred_params': pred_params,
                'input_data': x_sample[0]
            })
        
        self.logger.info(f"模型评估完成: {metrics}")
        return metrics
    
    def visualize_results(self, save_plots: bool = True):
        """
        可视化实验结果
        
        Args:
            save_plots: 是否保存图片
        """
        self.logger.info("生成可视化结果...")
        
        # 设置绘图样式
        plt.style.use(self.config.visualization_params['style'])
        sns.set_palette(self.config.visualization_params['color_palette'])
        
        # 1. 训练历史
        self._plot_training_history(save_plots)
        
        # 2. 参数预测对比
        self._plot_parameter_predictions(save_plots)
        
        # 3. 函数重构对比
        self._plot_function_reconstruction(save_plots)
        
        self.logger.info("可视化完成")
    
    def _plot_training_history(self, save_plots: bool):
        """
        绘制训练历史
        """
        if not self.results['training_history']:
            return
        
        history = self.results['training_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('参数拟合模型训练历史', fontsize=16)
        
        # 总损失
        axes[0, 0].plot(history['train_loss'], label='训练损失', alpha=0.8)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='验证损失', alpha=0.8)
        axes[0, 0].set_title('总损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 参数损失
        if 'train_param_loss' in history:
            axes[0, 1].plot(history['train_param_loss'], label='训练参数损失', alpha=0.8)
            if 'val_param_loss' in history:
                axes[0, 1].plot(history['val_param_loss'], label='验证参数损失', alpha=0.8)
            axes[0, 1].set_title('参数损失')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Parameter Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 重构损失
        if 'train_recon_loss' in history:
            axes[1, 0].plot(history['train_recon_loss'], label='训练重构损失', alpha=0.8)
            if 'val_recon_loss' in history:
                axes[1, 0].plot(history['val_recon_loss'], label='验证重构损失', alpha=0.8)
            axes[1, 0].set_title('重构损失')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reconstruction Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], alpha=0.8)
            axes[1, 1].set_title('学习率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.config.paths['plots'], 'parameter_training_history.png'),
                       dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _plot_parameter_predictions(self, save_plots: bool):
        """
        绘制参数预测对比
        """
        if not self.results['best_predictions']:
            return
        
        num_examples = min(6, len(self.results['best_predictions']))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('参数预测对比 (前6个样例)', fontsize=16)
        
        axes = axes.flatten()
        
        for i in range(num_examples):
            pred_data = self.results['best_predictions'][i]
            true_params = pred_data['true_params']
            pred_params = pred_data['pred_params']
            
            # 提取参数名称和值
            param_names = list(true_params.keys())
            true_values = [true_params[name] for name in param_names]
            pred_values = [pred_params[name] for name in param_names]
            
            x_pos = np.arange(len(param_names))
            width = 0.35
            
            axes[i].bar(x_pos - width/2, true_values, width, label='真实值', alpha=0.8)
            axes[i].bar(x_pos + width/2, pred_values, width, label='预测值', alpha=0.8)
            
            axes[i].set_title(f'样例 {i+1}')
            axes[i].set_xlabel('参数')
            axes[i].set_ylabel('值')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(param_names, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.config.paths['plots'], 'parameter_predictions.png'),
                       dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def _plot_function_reconstruction(self, save_plots: bool):
        """
        绘制函数重构对比
        """
        if not self.results['best_predictions']:
            return
        
        num_examples = min(4, len(self.results['best_predictions']))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('函数重构对比', fontsize=16)
        
        axes = axes.flatten()
        
        num_points = self.config.freq2_params['num_data_points']
        x_range = self.config.freq2_params['x_range']
        x_plot = np.linspace(x_range[0], x_range[1], num_points)
        
        for i in range(num_examples):
            pred_data = self.results['best_predictions'][i]
            true_params = pred_data['true_params']
            pred_params = pred_data['pred_params']
            input_data = pred_data['input_data']
            
            # 提取输入的x和y数据
            x_input = input_data[:num_points]
            y_input = input_data[num_points:]
            
            # 重构函数
            y_true_recon = self.data_generator.generate_frequency_function(x_plot, true_params)
            y_pred_recon = self.data_generator.generate_frequency_function(x_plot, pred_params)
            
            axes[i].plot(x_plot, y_true_recon, 'b-', label='真实函数', linewidth=2, alpha=0.8)
            axes[i].plot(x_plot, y_pred_recon, 'r--', label='预测函数', linewidth=2, alpha=0.8)
            axes[i].scatter(x_input, y_input, c='green', s=20, alpha=0.6, label='输入数据点')
            
            axes[i].set_title(f'样例 {i+1}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.config.paths['plots'], 'function_reconstruction.png'),
                       dpi=self.config.visualization_params['dpi'], bbox_inches='tight')
        plt.show()
    
    def save_experiment_results(self):
        """
        保存实验结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.config.paths['results'], 
                                   f'parameter_fitting_results_{timestamp}.json')
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = self._make_json_serializable(self.results)
        
        save_results(serializable_results, results_file)
        self.logger.info(f"实验结果已保存到: {results_file}")
        
        # 保存模型
        if self.trainer:
            model_file = os.path.join(self.config.paths['models'], 
                                     f'parameter_model_{timestamp}.pth')
            self.trainer.save_checkpoint(model_file)
            self.logger.info(f"模型已保存到: {model_file}")
    
    def _make_json_serializable(self, obj):
        """
        将对象转换为JSON可序列化格式
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def run_complete_experiment(self, num_train_samples: int = 1000, 
                               num_test_samples: int = 200):
        """
        运行完整的参数拟合实验
        
        Args:
            num_train_samples: 训练样本数量
            num_test_samples: 测试样本数量
        """
        self.logger.info("开始完整的参数拟合实验...")
        
        try:
            # 1. 生成数据集
            X_all, y_all = self.generate_training_dataset(num_train_samples + num_test_samples)
            
            # 2. 划分训练集和测试集
            X_train = X_all[:num_train_samples]
            y_train = y_all[:num_train_samples]
            X_test = X_all[num_train_samples:]
            y_test = y_all[num_train_samples:]
            
            # 3. 进一步划分训练集和验证集
            val_split = self.config.freq2_params['validation_split']
            val_size = int(len(X_train) * val_split)
            
            X_val = X_train[:val_size]
            y_val = y_train[:val_size]
            X_train = X_train[val_size:]
            y_train = y_train[val_size:]
            
            self.logger.info(f"数据集划分: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
            
            # 4. 训练模型
            self.train_model(X_train, y_train, X_val, y_val)
            
            # 5. 评估模型
            self.evaluate_model(X_test, y_test)
            
            # 6. 可视化结果
            self.visualize_results()
            
            # 7. 保存结果
            self.save_experiment_results()
            
            self.logger.info("参数拟合实验完成!")
            
        except Exception as e:
            self.logger.error(f"实验过程中发生错误: {str(e)}")
            raise

def main():
    """
    主函数
    """
    print("=" * 60)
    print("频率特性实验 - 第二步：频域参数拟合")
    print("=" * 60)
    
    # 初始化配置
    config = Config()
    
    # 创建实验实例
    experiment = ParameterFittingExperiment(config)
    
    # 运行完整实验
    experiment.run_complete_experiment(
        num_train_samples=800,
        num_test_samples=200
    )
    
    print("\n实验完成! 请查看生成的图表和结果文件。")

if __name__ == "__main__":
    main()