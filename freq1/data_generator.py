"""频率特性数据生成器

本模块实现了频率函数的构造和数据采样功能。
根据给定参数生成具有明显频率特征的训练数据。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
import os
from config import ExperimentConfig


class FrequencyDataGenerator:
    """频率数据生成器
    
    根据给定的频率函数参数生成训练和测试数据。
    函数形式：f(x) = a0 + a1*sin(x+b1) + a2*sin(2x+b2) + a3*sin(3x+b3) + ...
    """
    
    def __init__(self, config: ExperimentConfig):
        """初始化数据生成器
        
        Args:
            config: 实验配置对象
        """
        self.config = config
        self.coefficients = config.data_params['coefficients']
        self.frequency_components = config.get_frequency_components()
        
        # 设置随机种子
        np.random.seed(config.data_params['random_seed'])
    
    def frequency_function(self, x: np.ndarray) -> np.ndarray:
        """计算频率函数值
        
        Args:
            x: 输入值数组
            
        Returns:
            np.ndarray: 函数值数组
        """
        # 常数项
        y = np.full_like(x, self.coefficients['a0'])
        
        # 添加各频率成分
        for amplitude, phase, freq in self.frequency_components:
            y += amplitude * np.sin(freq * x + phase)
        
        return y
    
    def add_noise(self, y: np.ndarray, noise_level: float) -> np.ndarray:
        """添加高斯噪声
        
        Args:
            y: 原始函数值
            noise_level: 噪声标准差
            
        Returns:
            np.ndarray: 添加噪声后的函数值
        """
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, y.shape)
            return y + noise
        return y
    
    def generate_samples(self, 
                        num_samples: int, 
                        x_range: Tuple[float, float],
                        noise_level: float = 0.0,
                        sampling_method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """生成训练样本
        
        Args:
            num_samples: 样本数量
            x_range: 采样范围 (min, max)
            noise_level: 噪声水平
            sampling_method: 采样方法 ('uniform', 'random')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (x_samples, y_samples)
        """
        x_min, x_max = x_range
        
        if sampling_method == 'uniform':
            # 均匀采样
            x_samples = np.linspace(x_min, x_max, num_samples)
        elif sampling_method == 'random':
            # 随机采样
            x_samples = np.random.uniform(x_min, x_max, num_samples)
            x_samples = np.sort(x_samples)  # 排序便于可视化
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # 计算函数值
        y_clean = self.frequency_function(x_samples)
        
        # 添加噪声
        y_samples = self.add_noise(y_clean, noise_level)
        
        return x_samples.reshape(-1, 1), y_samples.reshape(-1, 1)
    
    def generate_train_test_data(self) -> Dict[str, np.ndarray]:
        """生成训练和测试数据集
        
        Returns:
            Dict[str, np.ndarray]: 包含训练和测试数据的字典
        """
        data_params = self.config.data_params
        
        # 生成训练数据
        x_train, y_train = self.generate_samples(
            num_samples=data_params['num_samples'],
            x_range=data_params['x_range'],
            noise_level=data_params['noise_level'],
            sampling_method='uniform'
        )
        
        # 生成测试数据（无噪声，用于评估）
        x_test, y_test = self.generate_samples(
            num_samples=data_params['num_test_samples'],
            x_range=data_params['x_range'],
            noise_level=0.0,  # 测试数据不加噪声
            sampling_method='uniform'
        )
        
        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
    
    def generate_high_resolution_data(self, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """生成高分辨率数据用于可视化
        
        Args:
            num_points: 数据点数量
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (x, y) 高分辨率数据
        """
        x_range = self.config.data_params['x_range']
        x_high_res = np.linspace(x_range[0], x_range[1], num_points)
        y_high_res = self.frequency_function(x_high_res)
        
        return x_high_res.reshape(-1, 1), y_high_res.reshape(-1, 1)
    
    def analyze_frequency_components(self) -> Dict[str, Any]:
        """分析频率成分
        
        Returns:
            Dict[str, Any]: 频率分析结果
        """
        analysis = {
            'constant_term': self.coefficients['a0'],
            'frequency_components': [],
            'total_amplitude': 0.0,
            'dominant_frequency': None,
            'max_amplitude': 0.0
        }
        
        max_amp = 0.0
        dominant_freq = None
        
        for amplitude, phase, freq in self.frequency_components:
            component_info = {
                'frequency': freq,
                'amplitude': amplitude,
                'phase': phase,
                'phase_degrees': np.degrees(phase),
                'contribution_ratio': 0.0  # 稍后计算
            }
            analysis['frequency_components'].append(component_info)
            analysis['total_amplitude'] += abs(amplitude)
            
            if abs(amplitude) > max_amp:
                max_amp = abs(amplitude)
                dominant_freq = freq
        
        analysis['max_amplitude'] = max_amp
        analysis['dominant_frequency'] = dominant_freq
        
        # 计算贡献比例
        if analysis['total_amplitude'] > 0:
            for component in analysis['frequency_components']:
                component['contribution_ratio'] = abs(component['amplitude']) / analysis['total_amplitude']
        
        return analysis
    
    def save_data(self, data: Dict[str, np.ndarray], filepath: str) -> None:
        """保存数据到文件
        
        Args:
            data: 数据字典
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, **data)
        print(f"数据已保存到: {filepath}")
    
    def load_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """从文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            Dict[str, np.ndarray]: 加载的数据
        """
        data = np.load(filepath)
        return {key: data[key] for key in data.files}
    
    def plot_function_and_samples(self, 
                                 data: Dict[str, np.ndarray], 
                                 save_path: Optional[str] = None) -> None:
        """绘制原函数和采样数据
        
        Args:
            data: 数据字典
            save_path: 保存路径（可选）
        """
        # 生成高分辨率数据用于绘制原函数
        x_high_res, y_high_res = self.generate_high_resolution_data()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.visualization_params['figure_size'])
        colors = self.config.visualization_params['colors']
        
        # 上图：原函数和训练数据
        ax1.plot(x_high_res.flatten(), y_high_res.flatten(), 
                label='原函数', color=colors['original'], linewidth=2)
        ax1.scatter(data['x_train'].flatten(), data['y_train'].flatten(), 
                   label='训练样本', color=colors['samples'], alpha=0.6, s=20)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('频率函数与训练样本')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：测试数据
        ax2.plot(x_high_res.flatten(), y_high_res.flatten(), 
                label='原函数', color=colors['original'], linewidth=2)
        ax2.scatter(data['x_test'].flatten(), data['y_test'].flatten(), 
                   label='测试样本', color=colors['prediction'], alpha=0.8, s=20)
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('频率函数与测试样本')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.config.visualization_params['dpi'], 
                       bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数：演示数据生成器的使用"""
    # 创建配置和数据生成器
    config = ExperimentConfig()
    generator = FrequencyDataGenerator(config)
    
    # 分析频率成分
    analysis = generator.analyze_frequency_components()
    print("频率成分分析:")
    print(f"常数项: {analysis['constant_term']:.3f}")
    print(f"主导频率: {analysis['dominant_frequency']}")
    print(f"最大幅度: {analysis['max_amplitude']:.3f}")
    print("\n各频率成分:")
    for comp in analysis['frequency_components']:
        print(f"  频率 {comp['frequency']}: 幅度={comp['amplitude']:.3f}, "
              f"相位={comp['phase_degrees']:.1f}°, 贡献={comp['contribution_ratio']:.1%}")
    
    # 生成数据
    print("\n生成训练和测试数据...")
    data = generator.generate_train_test_data()
    
    print(f"训练样本数: {len(data['x_train'])}")
    print(f"测试样本数: {len(data['x_test'])}")
    print(f"数据范围: [{data['x_train'].min():.2f}, {data['x_train'].max():.2f}]")
    print(f"函数值范围: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")
    
    # 保存数据
    os.makedirs('data', exist_ok=True)
    generator.save_data(data, 'data/frequency_data.npz')
    
    # 可视化
    print("\n生成可视化图像...")
    os.makedirs('results/plots', exist_ok=True)
    generator.plot_function_and_samples(data, 'results/plots/data_generation.png')


if __name__ == '__main__':
    main()