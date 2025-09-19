"""
泛化实验主脚本
整合所有模块，进行完整的深度神经网络泛化现象研究
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from data_generator import create_experiment_data, DataGenerator
from model import ModelFactory, create_overparameterized_configs
from trainer import Trainer, create_data_loaders, TrainingHistory
from evaluator import ModelEvaluator, Visualizer


class GeneralizationExperiment:
    """泛化实验类"""
    
    def __init__(self, 
                 experiment_name: str = "generalization_study",
                 save_dir: str = "./results",
                 device: str = None):
        """
        初始化实验
        
        Args:
            experiment_name: 实验名称
            save_dir: 保存目录
            device: 计算设备
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 设备选择
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 创建子目录
        self.models_dir = self.save_dir / "models"
        self.plots_dir = self.save_dir / "plots"
        self.histories_dir = self.save_dir / "histories"
        
        for dir_path in [self.models_dir, self.plots_dir, self.histories_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 实验结果存储
        self.results = {}
        self.histories = {}
        self.models = {}
        
        # 可视化工具
        self.visualizer = Visualizer()
    
    def setup_experiment(self,
                        polynomial_degree: int = 3,
                        n_samples: int = 30,
                        noise_std: float = 0.1,
                        test_range: Tuple[float, float] = (-3, 3),
                        random_seed: int = 42):
        """
        设置实验参数和数据
        
        Args:
            polynomial_degree: 多项式度数
            n_samples: 训练样本数量
            noise_std: 噪声标准差
            test_range: 测试范围
            random_seed: 随机种子
        """
        print(f"Setting up experiment with {n_samples} samples...")
        
        # 创建实验数据
        self.data_generator, self.x_train, self.y_train, self.x_test, self.y_test = create_experiment_data(
            degree=polynomial_degree,
            n_samples=n_samples,
            noise_std=noise_std,
            random_seed=random_seed
        )
        
        # 扩展测试范围
        self.x_test_extended = np.linspace(test_range[0], test_range[1], 1000).reshape(-1, 1)
        self.y_test_extended = self.data_generator.polynomial.evaluate(self.x_test_extended)
        
        # 保存实验配置
        self.config = {
            'polynomial_degree': polynomial_degree,
            'n_samples': n_samples,
            'noise_std': noise_std,
            'test_range': test_range,
            'random_seed': random_seed,
            'polynomial_formula': self.data_generator.polynomial.get_formula(),
            'device': self.device
        }
        
        print(f"Polynomial: {self.config['polynomial_formula']}")
        print(f"Training samples: {n_samples}")
        print(f"Test range: {test_range}")
        
        # 可视化原始数据
        self.data_generator.visualize_data(
            self.x_train, self.y_train, 
            self.x_test_extended, self.y_test_extended,
            title="Experiment Data"
        )
    
    def create_model_configurations(self, max_layers: int = 6) -> List[Dict]:
        """
        创建模型配置
        
        Args:
            max_layers: 最大层数
            
        Returns:
            模型配置列表
        """
        configs = []
        
        # 创建不同层数的过参数化配置
        layer_configs = [
            {'layers': 2, 'width': 128, 'name': 'L2_W128'},
            {'layers': 3, 'width': 64, 'name': 'L3_W64'},
            {'layers': 4, 'width': 48, 'name': 'L4_W48'},
            {'layers': 5, 'width': 32, 'name': 'L5_W32'},
            {'layers': 6, 'width': 24, 'name': 'L6_W24'},
        ]
        
        for config in layer_configs:
            if config['layers'] <= max_layers:
                model_config = {
                    'name': config['name'],
                    'hidden_dims': [config['width']] * config['layers'],
                    'activation': 'relu',
                    'dropout_rate': 0.0,
                    'use_batch_norm': False
                }
                configs.append(model_config)
        
        # 添加一些变宽度的配置
        configs.extend([
            {
                'name': 'L3_Decreasing',
                'hidden_dims': [128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.0,
                'use_batch_norm': False
            },
            {
                'name': 'L4_Pyramid',
                'hidden_dims': [256, 128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.0,
                'use_batch_norm': False
            }
        ])
        
        return configs
    
    def train_models(self,
                    model_configs: List[Dict],
                    num_epochs: int = 2000,
                    learning_rate: float = 0.001,
                    batch_size: int = 16,
                    early_stopping_patience: int = 200,
                    print_every: int = 200):
        """
        训练所有模型
        
        Args:
            model_configs: 模型配置列表
            num_epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批大小
            early_stopping_patience: 早停耐心值
            print_every: 打印间隔
        """
        print(f"\nTraining {len(model_configs)} models...")
        print("=" * 60)
        
        # 创建数据加载器
        train_loader, _ = create_data_loaders(
            self.x_train, self.y_train, 
            batch_size=batch_size, shuffle=True
        )
        
        for i, config in enumerate(model_configs):
            print(f"\nTraining model {i+1}/{len(model_configs)}: {config['name']}")
            print("-" * 40)
            
            # 创建模型
            model = ModelFactory.create_model(config)
            print(f"Model parameters: {model.get_num_parameters()}")
            print(f"Overparameterized: {model.get_num_parameters() > len(self.x_train)}")
            
            # 创建训练器
            trainer = Trainer(model, device=self.device, save_dir=str(self.histories_dir))
            
            # 训练模型
            history = trainer.train(
                train_loader=train_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                print_every=print_every,
                save_best=True
            )
            
            # 保存结果
            self.models[config['name']] = model
            self.histories[config['name']] = history
            
            # 保存模型和历史
            model_path = self.models_dir / f"{config['name']}_model.pth"
            history_path = self.histories_dir / f"{config['name']}_history.json"
            
            trainer.save_model(str(model_path))
            history.save(str(history_path))
            
            print(f"Model saved to: {model_path}")
            print(f"Final training loss: {history.train_losses[-1]:.6f}")
    
    def evaluate_models(self) -> Dict[str, Dict]:
        """
        评估所有训练好的模型
        
        Returns:
            评估结果字典
        """
        print(f"\nEvaluating {len(self.models)} models...")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            evaluator = ModelEvaluator(model, device=self.device)
            
            # 在扩展测试集上评估
            model_results = evaluator.evaluate_on_data(
                self.x_test_extended, self.y_test_extended,
                self.x_train, self.y_train
            )
            
            # 添加模型信息
            model_results['model_name'] = name
            model_results['hidden_dims'] = model.hidden_dims
            model_results['num_layers'] = len(model.hidden_dims)
            
            results[name] = model_results
            
            # 打印关键指标
            print(f"  Test MSE: {model_results['test_mse']:.6f}")
            print(f"  Test R²: {model_results['test_r2']:.4f}")
            print(f"  Overfitting Ratio: {model_results['overfitting_ratio']:.4f}")
            print(f"  Smoothness Score: {model_results['oscillation_metrics']['smoothness_score']:.4f}")
        
        self.results = results
        return results
    
    def generate_visualizations(self):
        """生成所有可视化图表"""
        print(f"\nGenerating visualizations...")
        print("=" * 60)
        
        # 1. 训练历史
        print("Plotting training histories...")
        self.visualizer.plot_training_history(
            self.histories,
            save_path=str(self.plots_dir / "training_histories.png")
        )
        
        # 2. 函数拟合结果
        print("Plotting function fitting results...")
        models_and_names = [(model, name) for name, model in self.models.items()]
        self.visualizer.plot_function_fitting(
            self.data_generator,
            models_and_names,
            self.x_train, self.y_train,
            self.x_test_extended, self.y_test_extended,
            save_path=str(self.plots_dir / "function_fitting.png")
        )
        
        # 3. 震荡分析
        print("Plotting oscillation analysis...")
        self.visualizer.plot_oscillation_analysis(
            models_and_names,
            self.x_test_extended,
            save_path=str(self.plots_dir / "oscillation_analysis.png")
        )
        
        # 4. 泛化分析
        print("Plotting generalization analysis...")
        self.visualizer.plot_generalization_analysis(
            self.results,
            save_path=str(self.plots_dir / "generalization_analysis.png")
        )
        
        print("All visualizations saved to:", self.plots_dir)
    
    def save_experiment_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_file = self.save_dir / f"experiment_results_{timestamp}.json"
        
        experiment_data = {
            'experiment_name': self.experiment_name,
            'timestamp': timestamp,
            'config': self.config,
            'results': self.results,
            'summary': self.generate_summary()
        }
        
        with open(results_file, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        print(f"Experiment results saved to: {results_file}")
        
        return results_file
    
    def generate_summary(self) -> Dict:
        """生成实验总结"""
        if not self.results:
            return {}
        
        # 统计信息
        model_names = list(self.results.keys())
        test_mses = [self.results[name]['test_mse'] for name in model_names]
        param_counts = [self.results[name]['model_params'] for name in model_names]
        r2_scores = [self.results[name]['test_r2'] for name in model_names]
        overfitting_ratios = [self.results[name]['overfitting_ratio'] for name in model_names]
        
        summary = {
            'total_models': len(model_names),
            'best_model': {
                'name': model_names[np.argmin(test_mses)],
                'test_mse': min(test_mses),
                'r2_score': r2_scores[np.argmin(test_mses)]
            },
            'most_overparameterized': {
                'name': model_names[np.argmax(param_counts)],
                'parameters': max(param_counts),
                'samples_ratio': max(param_counts) / len(self.x_train)
            },
            'average_overfitting_ratio': np.mean(overfitting_ratios),
            'models_with_good_generalization': sum(1 for ratio in overfitting_ratios if ratio < 2.0),
            'parameter_range': {
                'min': min(param_counts),
                'max': max(param_counts),
                'mean': np.mean(param_counts)
            }
        }
        
        return summary
    
    def run_complete_experiment(self,
                              polynomial_degree: int = 3,
                              n_samples: int = 30,
                              noise_std: float = 0.1,
                              max_layers: int = 6,
                              num_epochs: int = 2000,
                              learning_rate: float = 0.001):
        """
        运行完整的实验流程
        
        Args:
            polynomial_degree: 多项式度数
            n_samples: 训练样本数量
            noise_std: 噪声标准差
            max_layers: 最大层数
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        print("=" * 80)
        print("GENERALIZATION MYSTERY EXPERIMENT")
        print("=" * 80)
        
        # 1. 设置实验
        self.setup_experiment(
            polynomial_degree=polynomial_degree,
            n_samples=n_samples,
            noise_std=noise_std
        )
        
        # 2. 创建模型配置
        model_configs = self.create_model_configurations(max_layers=max_layers)
        
        # 打印模型配置信息
        print(f"\nModel configurations:")
        ModelFactory.print_model_comparison(model_configs, n_samples)
        
        # 3. 训练模型
        self.train_models(
            model_configs=model_configs,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # 4. 评估模型
        self.evaluate_models()
        
        # 5. 生成可视化
        self.generate_visualizations()
        
        # 6. 保存结果
        results_file = self.save_experiment_results()
        
        # 7. 打印总结
        self.print_experiment_summary()
        
        print("=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {self.save_dir}")
        print("=" * 80)
        
        return results_file
    
    def print_experiment_summary(self):
        """打印实验总结"""
        summary = self.generate_summary()
        
        print(f"\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        print(f"Total models trained: {summary['total_models']}")
        print(f"Training samples: {len(self.x_train)}")
        print(f"Polynomial: {self.config['polynomial_formula']}")
        
        print(f"\nBest performing model:")
        print(f"  Name: {summary['best_model']['name']}")
        print(f"  Test MSE: {summary['best_model']['test_mse']:.6f}")
        print(f"  R² Score: {summary['best_model']['r2_score']:.4f}")
        
        print(f"\nMost overparameterized model:")
        print(f"  Name: {summary['most_overparameterized']['name']}")
        print(f"  Parameters: {summary['most_overparameterized']['parameters']}")
        print(f"  Param/Sample ratio: {summary['most_overparameterized']['samples_ratio']:.1f}")
        
        print(f"\nGeneralization analysis:")
        print(f"  Average overfitting ratio: {summary['average_overfitting_ratio']:.2f}")
        print(f"  Models with good generalization (ratio < 2.0): {summary['models_with_good_generalization']}/{summary['total_models']}")
        
        print(f"\nParameter statistics:")
        print(f"  Range: {summary['parameter_range']['min']} - {summary['parameter_range']['max']}")
        print(f"  Average: {summary['parameter_range']['mean']:.0f}")


if __name__ == "__main__":
    # 运行完整实验
    experiment = GeneralizationExperiment(
        experiment_name="deep_generalization_mystery",
        save_dir="./results"
    )
    
    # 运行实验
    results_file = experiment.run_complete_experiment(
        polynomial_degree=3,
        n_samples=30,
        noise_std=0.1,
        max_layers=6,
        num_epochs=1500,
        learning_rate=0.001
    )