"""主实验脚本

本脚本整合了数据生成、模型训练、评估和可视化的完整实验流程。
提供了灵活的实验配置和批量实验功能。
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import ExperimentConfig
from data_generator import FrequencyDataGenerator
from model import TwoLayerNet, create_model
from train import run_experiment, load_model
from visualize import ExperimentVisualizer


def create_experiment_variants() -> List[Dict[str, Any]]:
    """创建不同的实验变体
    
    Returns:
        List[Dict[str, Any]]: 实验配置变体列表
    """
    variants = []
    
    # 基础实验
    variants.append({
        'name': 'baseline',
        'description': '基础实验：标准两层网络',
        'config_updates': {}
    })
    
    # 不同隐藏层大小
    for hidden_dim in [32, 64, 128, 256]:
        variants.append({
            'name': f'hidden_{hidden_dim}',
            'description': f'隐藏层维度: {hidden_dim}',
            'config_updates': {
                'model_params': {'hidden_dim': hidden_dim}
            }
        })
    
    # 不同学习率
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        variants.append({
            'name': f'lr_{lr:.4f}'.replace('.', '_'),
            'description': f'学习率: {lr}',
            'config_updates': {
                'training_params': {'learning_rate': lr}
            }
        })
    
    # 不同激活函数
    for activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']:
        variants.append({
            'name': f'activation_{activation}',
            'description': f'激活函数: {activation}',
            'config_updates': {
                'model_params': {'activation': activation}
            }
        })
    
    # 不同噪声水平
    for noise in [0.0, 0.01, 0.05, 0.1, 0.2]:
        variants.append({
            'name': f'noise_{noise:.2f}'.replace('.', '_'),
            'description': f'噪声水平: {noise}',
            'config_updates': {
                'data_params': {'noise_level': noise}
            }
        })
    
    return variants


def run_single_experiment(variant: Dict[str, Any], 
                         base_config: ExperimentConfig,
                         save_results: bool = True) -> Dict[str, Any]:
    """运行单个实验
    
    Args:
        variant: 实验变体配置
        base_config: 基础配置
        save_results: 是否保存结果
        
    Returns:
        Dict[str, Any]: 实验结果
    """
    print(f"\n{'='*60}")
    print(f"运行实验: {variant['name']}")
    print(f"描述: {variant['description']}")
    print(f"{'='*60}")
    
    # 创建实验配置
    config = ExperimentConfig()
    if variant['config_updates']:
        config.update_config(variant['config_updates'])
    
    # 运行实验
    experiment_name = f"freq1_{variant['name']}"
    results = run_experiment(
        config=config,
        save_results=save_results,
        experiment_name=experiment_name
    )
    
    # 添加变体信息
    results['variant_info'] = variant
    
    return results


def run_batch_experiments(variants: List[Dict[str, Any]], 
                         base_config: ExperimentConfig,
                         save_results: bool = True) -> List[Dict[str, Any]]:
    """批量运行实验
    
    Args:
        variants: 实验变体列表
        base_config: 基础配置
        save_results: 是否保存结果
        
    Returns:
        List[Dict[str, Any]]: 所有实验结果
    """
    all_results = []
    
    print(f"开始批量实验，共 {len(variants)} 个变体")
    
    for i, variant in enumerate(variants, 1):
        print(f"\n进度: {i}/{len(variants)}")
        
        try:
            results = run_single_experiment(variant, base_config, save_results)
            all_results.append(results)
            
            # 打印关键指标
            r2_score = results['evaluation_results']['r2_score']
            rmse = results['evaluation_results']['rmse']
            epochs = results['training_results']['total_epochs']
            print(f"结果: R²={r2_score:.4f}, RMSE={rmse:.4f}, Epochs={epochs}")
            
        except Exception as e:
            print(f"实验 {variant['name']} 失败: {str(e)}")
            continue
    
    return all_results


def compare_experiments(results_list: List[Dict[str, Any]], 
                       save_path: Optional[str] = None) -> None:
    """比较实验结果
    
    Args:
        results_list: 实验结果列表
        save_path: 保存路径
    """
    if not results_list:
        print("没有实验结果可比较")
        return
    
    print("\n" + "="*80)
    print("实验结果比较")
    print("="*80)
    
    # 创建比较表格
    headers = ['实验名称', '描述', 'R²', 'RMSE', 'MAE', '训练轮数', '最终损失']
    
    print(f"{headers[0]:<20} {headers[1]:<30} {headers[2]:<8} {headers[3]:<8} {headers[4]:<8} {headers[5]:<8} {headers[6]:<12}")
    print("-" * 100)
    
    # 排序：按R²分数降序
    sorted_results = sorted(results_list, 
                           key=lambda x: x['evaluation_results']['r2_score'], 
                           reverse=True)
    
    for result in sorted_results:
        name = result['variant_info']['name']
        desc = result['variant_info']['description']
        eval_results = result['evaluation_results']
        train_results = result['training_results']
        
        r2 = eval_results['r2_score']
        rmse = eval_results['rmse']
        mae = eval_results['mae']
        epochs = train_results['total_epochs']
        final_loss = train_results['final_train_loss']
        
        print(f"{name:<20} {desc:<30} {r2:<8.4f} {rmse:<8.4f} {mae:<8.4f} {epochs:<8} {final_loss:<12.6f}")
    
    # 找出最佳结果
    best_result = sorted_results[0]
    print(f"\n最佳实验: {best_result['variant_info']['name']}")
    print(f"R² Score: {best_result['evaluation_results']['r2_score']:.6f}")
    print(f"RMSE: {best_result['evaluation_results']['rmse']:.6f}")
    
    # 保存比较结果
    if save_path:
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'best_experiment': best_result['variant_info']['name'],
            'results_summary': []
        }
        
        for result in sorted_results:
            summary = {
                'name': result['variant_info']['name'],
                'description': result['variant_info']['description'],
                'metrics': result['evaluation_results'],
                'training_info': {
                    'epochs': result['training_results']['total_epochs'],
                    'final_loss': result['training_results']['final_train_loss']
                }
            }
            comparison_data['results_summary'].append(summary)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n比较结果已保存到: {save_path}")


def generate_experiment_report(experiment_name: str) -> None:
    """生成实验报告
    
    Args:
        experiment_name: 实验名称
    """
    model_path = f'results/{experiment_name}/model.pth'
    if not os.path.exists(model_path):
        print(f"未找到实验 {experiment_name} 的模型文件")
        return
    
    print(f"生成实验 {experiment_name} 的可视化报告...")
    
    # 加载配置和模型
    config = ExperimentConfig()
    model = load_model(model_path, config)
    
    # 生成数据
    generator = FrequencyDataGenerator(config)
    data = generator.generate_train_test_data()
    
    # 加载训练历史
    history_path = f'results/{experiment_name}/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    else:
        training_history = {'train_loss': [], 'val_loss': []}
    
    # 创建可视化器并生成报告
    visualizer = ExperimentVisualizer(config)
    report_dir = f'results/{experiment_name}/plots'
    visualizer.create_comprehensive_report(
        generator, model, data, training_history, report_dir
    )
    
    print(f"实验报告已生成: {report_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='频率特性学习实验')
    parser.add_argument('--mode', choices=['single', 'batch', 'compare', 'report'], 
                       default='single', help='运行模式')
    parser.add_argument('--variant', type=str, help='实验变体名称（single模式）')
    parser.add_argument('--experiment', type=str, help='实验名称（report模式）')
    parser.add_argument('--no-save', action='store_true', help='不保存结果')
    
    args = parser.parse_args()
    
    # 创建基础配置
    base_config = ExperimentConfig()
    
    if args.mode == 'single':
        # 单个实验
        if args.variant:
            # 查找指定变体
            variants = create_experiment_variants()
            variant = next((v for v in variants if v['name'] == args.variant), None)
            if variant is None:
                print(f"未找到变体: {args.variant}")
                print("可用变体:")
                for v in variants[:10]:  # 只显示前10个
                    print(f"  {v['name']}: {v['description']}")
                return
        else:
            # 默认基础实验
            variant = {
                'name': 'baseline',
                'description': '基础实验',
                'config_updates': {}
            }
        
        run_single_experiment(variant, base_config, not args.no_save)
    
    elif args.mode == 'batch':
        # 批量实验
        variants = create_experiment_variants()
        print(f"将运行 {len(variants)} 个实验变体")
        
        # 只运行部分变体以节省时间
        selected_variants = [
            variants[0],  # baseline
            *[v for v in variants if 'hidden' in v['name']][:3],  # 不同隐藏层大小
            *[v for v in variants if 'lr' in v['name']][:3],      # 不同学习率
            *[v for v in variants if 'noise' in v['name']][:3],   # 不同噪声水平
        ]
        
        results = run_batch_experiments(selected_variants, base_config, not args.no_save)
        
        # 比较结果
        if results:
            compare_experiments(results, 'results/batch_comparison.json')
    
    elif args.mode == 'compare':
        # 比较现有实验结果
        results_dir = 'results'
        if not os.path.exists(results_dir):
            print("未找到实验结果目录")
            return
        
        # 加载所有实验结果
        all_results = []
        for exp_dir in os.listdir(results_dir):
            exp_path = os.path.join(results_dir, exp_dir)
            if os.path.isdir(exp_path):
                config_path = os.path.join(exp_path, 'experiment_config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            # 这里需要重新构造结果格式
                            print(f"找到实验: {exp_dir}")
                    except Exception as e:
                        print(f"加载实验 {exp_dir} 失败: {e}")
        
        print("请先运行批量实验以生成比较数据")
    
    elif args.mode == 'report':
        # 生成实验报告
        if not args.experiment:
            print("请指定实验名称")
            return
        
        generate_experiment_report(args.experiment)
    
    else:
        print("未知模式")


if __name__ == '__main__':
    main()