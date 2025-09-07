"""训练脚本

本脚本整合了数据生成、模型训练和评估的完整流程。
"""

import torch
import torch.utils.data as data
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple

from config import ExperimentConfig
from data_generator import FrequencyDataGenerator
from model import TwoLayerNet, ModelTrainer, create_model


class FrequencyDataset(data.Dataset):
    """频率数据集类"""
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        """初始化数据集
        
        Args:
            x_data: 输入数据
            y_data: 目标数据
        """
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
    
    def __len__(self) -> int:
        return len(self.x_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


def create_data_loaders(data: Dict[str, np.ndarray], 
                        config: ExperimentConfig) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """创建数据加载器
    
    Args:
        data: 数据字典
        config: 实验配置
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证、测试数据加载器
    """
    # 分割训练和验证数据
    val_split = config.training_params['validation_split']
    train_size = int(len(data['x_train']) * (1 - val_split))
    
    # 随机打乱训练数据
    indices = np.random.permutation(len(data['x_train']))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建数据集
    train_dataset = FrequencyDataset(
        data['x_train'][train_indices], 
        data['y_train'][train_indices]
    )
    val_dataset = FrequencyDataset(
        data['x_train'][val_indices], 
        data['y_train'][val_indices]
    )
    test_dataset = FrequencyDataset(
        data['x_test'], 
        data['y_test']
    )
    
    # 创建数据加载器
    batch_size = config.training_params['batch_size']
    
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False
    )
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader


def save_training_results(results: Dict[str, Any], 
                         model: TwoLayerNet, 
                         config: ExperimentConfig,
                         save_dir: str) -> None:
    """保存训练结果
    
    Args:
        results: 训练结果
        model: 训练好的模型
        config: 实验配置
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config.model_params,
        'training_results': results
    }, model_path)
    
    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(results['train_history'], f, indent=2, ensure_ascii=False)
    
    # 保存实验配置
    config_path = os.path.join(save_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    
    print(f"训练结果已保存到: {save_dir}")
    print(f"  模型文件: {model_path}")
    print(f"  训练历史: {history_path}")
    print(f"  实验配置: {config_path}")


def load_model(model_path: str, config: ExperimentConfig) -> TwoLayerNet:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        config: 实验配置
        
    Returns:
        TwoLayerNet: 加载的模型
    """
    model = TwoLayerNet(config)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"模型已从 {model_path} 加载")
    return model


def run_experiment(config: ExperimentConfig, 
                  save_results: bool = True,
                  experiment_name: str = None) -> Dict[str, Any]:
    """运行完整实验
    
    Args:
        config: 实验配置
        save_results: 是否保存结果
        experiment_name: 实验名称
        
    Returns:
        Dict[str, Any]: 实验结果
    """
    print("=" * 60)
    print("开始频率特性学习实验")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n1. 生成训练和测试数据...")
    generator = FrequencyDataGenerator(config)
    
    # 分析频率成分
    analysis = generator.analyze_frequency_components()
    print("\n频率成分分析:")
    print(f"  常数项: {analysis['constant_term']:.3f}")
    print(f"  主导频率: {analysis['dominant_frequency']}")
    print(f"  最大幅度: {analysis['max_amplitude']:.3f}")
    for comp in analysis['frequency_components']:
        print(f"  频率 {comp['frequency']}: 幅度={comp['amplitude']:.3f}, "
              f"相位={comp['phase_degrees']:.1f}°, 贡献={comp['contribution_ratio']:.1%}")
    
    # 生成数据
    data = generator.generate_train_test_data()
    print(f"\n数据生成完成:")
    print(f"  训练样本: {len(data['x_train'])}")
    print(f"  测试样本: {len(data['x_test'])}")
    print(f"  输入范围: [{data['x_train'].min():.2f}, {data['x_train'].max():.2f}]")
    print(f"  输出范围: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")
    
    # 2. 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(data, config)
    
    # 3. 创建和训练模型
    print("\n3. 创建神经网络模型...")
    model = create_model(config)
    
    print("\n4. 开始训练...")
    trainer = ModelTrainer(model, config)
    training_results = trainer.fit(train_loader, val_loader)
    
    print("\n训练完成!")
    print(f"  最终训练损失: {training_results['final_train_loss']:.6f}")
    if training_results['best_val_loss'] is not None:
        print(f"  最佳验证损失: {training_results['best_val_loss']:.6f}")
    print(f"  训练轮数: {training_results['total_epochs']}")
    
    # 4. 评估模型
    print("\n5. 评估模型性能...")
    eval_results = trainer.evaluate(test_loader)
    
    print("测试集评估结果:")
    print(f"  MSE: {eval_results['mse']:.6f}")
    print(f"  MAE: {eval_results['mae']:.6f}")
    print(f"  RMSE: {eval_results['rmse']:.6f}")
    print(f"  R² Score: {eval_results['r2_score']:.6f}")
    
    # 5. 整合结果
    experiment_results = {
        'experiment_info': {
            'name': experiment_name or f"freq_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict()
        },
        'frequency_analysis': analysis,
        'data_info': {
            'train_samples': len(data['x_train']),
            'test_samples': len(data['x_test']),
            'input_range': [float(data['x_train'].min()), float(data['x_train'].max())],
            'output_range': [float(data['y_train'].min()), float(data['y_train'].max())]
        },
        'training_results': training_results,
        'evaluation_results': eval_results,
        'model_info': model.get_model_info()
    }
    
    # 6. 保存结果
    if save_results:
        print("\n6. 保存实验结果...")
        save_dir = os.path.join('results', experiment_results['experiment_info']['name'])
        save_training_results(experiment_results, model, config, save_dir)
        
        # 保存数据
        data_dir = os.path.join(save_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        generator.save_data(data, os.path.join(data_dir, 'experiment_data.npz'))
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    
    return experiment_results


def main():
    """主函数"""
    # 创建配置
    config = ExperimentConfig()
    
    # 可以在这里修改配置参数
    # config.update_config({
    #     'training_params': {'num_epochs': 200, 'learning_rate': 0.01}
    # })
    
    # 运行实验
    results = run_experiment(
        config=config,
        save_results=True,
        experiment_name="freq1_baseline"
    )
    
    # 打印关键结果
    print("\n关键实验结果:")
    print(f"R² Score: {results['evaluation_results']['r2_score']:.4f}")
    print(f"RMSE: {results['evaluation_results']['rmse']:.4f}")
    print(f"训练轮数: {results['training_results']['total_epochs']}")


if __name__ == '__main__':
    main()