"""
数据生成模块
用于生成多项式函数和带噪声的训练数据
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class PolynomialGenerator:
    """多项式生成器"""
    
    def __init__(self, degree: int = 3, coefficients: Optional[List[float]] = None):
        """
        初始化多项式生成器
        
        Args:
            degree: 多项式的度数
            coefficients: 多项式系数，如果为None则随机生成
        """
        self.degree = degree
        if coefficients is None:
            # 随机生成系数，范围在[-2, 2]之间
            self.coefficients = np.random.uniform(-2, 2, degree + 1)
        else:
            self.coefficients = np.array(coefficients)
            self.degree = len(coefficients) - 1
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        计算多项式在给定点的值
        
        Args:
            x: 输入点
            
        Returns:
            多项式在x处的值
        """
        return np.polyval(self.coefficients, x)
    
    def get_formula(self) -> str:
        """
        获取多项式的字符串表示
        
        Returns:
            多项式公式字符串
        """
        terms = []
        for i, coeff in enumerate(self.coefficients):
            power = self.degree - i
            if abs(coeff) < 1e-6:
                continue
            
            if power == 0:
                terms.append(f"{coeff:.3f}")
            elif power == 1:
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff:.3f}x")
            else:
                if coeff == 1:
                    terms.append(f"x^{power}")
                elif coeff == -1:
                    terms.append(f"-x^{power}")
                else:
                    terms.append(f"{coeff:.3f}x^{power}")
        
        if not terms:
            return "0"
        
        formula = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                formula += f" {term}"
            else:
                formula += f" + {term}"
        
        return formula


class DataGenerator:
    """数据生成器"""
    
    def __init__(self, polynomial: PolynomialGenerator, noise_std: float = 0.1):
        """
        初始化数据生成器
        
        Args:
            polynomial: 多项式生成器
            noise_std: 高斯白噪声的标准差
        """
        self.polynomial = polynomial
        self.noise_std = noise_std
    
    def generate_training_data(self, 
                             n_samples: int, 
                             x_range: Tuple[float, float] = (-2, 2),
                             random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据
        
        Args:
            n_samples: 样本数量
            x_range: x的取值范围
            random_seed: 随机种子
            
        Returns:
            (x, y): 输入和输出数据
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 生成随机的x值
        x = np.random.uniform(x_range[0], x_range[1], n_samples)
        
        # 计算多项式的真实值
        y_true = self.polynomial.evaluate(x)
        
        # 添加高斯白噪声
        noise = np.random.normal(0, self.noise_std, n_samples)
        y = y_true + noise
        
        return x.reshape(-1, 1), y.reshape(-1, 1)
    
    def generate_test_data(self, 
                          x_range: Tuple[float, float] = (-3, 3),
                          n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成测试数据（连续的x值，用于观察拟合效果）
        
        Args:
            x_range: x的取值范围
            n_points: 点的数量
            
        Returns:
            (x, y): 输入和真实输出数据
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        y_true = self.polynomial.evaluate(x)
        
        return x.reshape(-1, 1), y_true.reshape(-1, 1)
    
    def visualize_data(self, 
                      x_train: np.ndarray, 
                      y_train: np.ndarray,
                      x_test: np.ndarray = None,
                      y_test: np.ndarray = None,
                      title: str = "Generated Data"):
        """
        可视化生成的数据
        
        Args:
            x_train: 训练数据的x
            y_train: 训练数据的y
            x_test: 测试数据的x
            y_test: 测试数据的y
            title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制训练数据点
        plt.scatter(x_train.flatten(), y_train.flatten(), 
                   alpha=0.6, color='blue', label='Training Data (with noise)')
        
        # 绘制真实函数曲线
        if x_test is not None and y_test is not None:
            plt.plot(x_test.flatten(), y_test.flatten(), 
                    'r-', linewidth=2, label='True Function')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{title}\nFormula: {self.polynomial.get_formula()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def create_experiment_data(degree: int = 3, 
                          n_samples: int = 50, 
                          noise_std: float = 0.1,
                          random_seed: int = 42) -> Tuple[DataGenerator, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建实验数据的便捷函数
    
    Args:
        degree: 多项式度数
        n_samples: 训练样本数量
        noise_std: 噪声标准差
        random_seed: 随机种子
        
    Returns:
        (data_generator, x_train, y_train, x_test, y_test)
    """
    # 设置随机种子以确保可重现性
    np.random.seed(random_seed)
    
    # 创建多项式生成器
    polynomial = PolynomialGenerator(degree=degree)
    
    # 创建数据生成器
    data_generator = DataGenerator(polynomial, noise_std=noise_std)
    
    # 生成训练数据
    x_train, y_train = data_generator.generate_training_data(
        n_samples=n_samples, 
        x_range=(-2, 2),
        random_seed=random_seed
    )
    
    # 生成测试数据（用于评估）
    x_test, y_test = data_generator.generate_test_data(
        x_range=(-3, 3),
        n_points=1000
    )
    
    return data_generator, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # 测试代码
    print("Testing data generator...")
    
    # 创建实验数据
    data_gen, x_train, y_train, x_test, y_test = create_experiment_data(
        degree=3, n_samples=30, noise_std=0.1
    )
    
    print(f"Training data shape: {x_train.shape}, {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, {y_test.shape}")
    print(f"Polynomial formula: {data_gen.polynomial.get_formula()}")
    
    # 可视化数据
    data_gen.visualize_data(x_train, y_train, x_test, y_test)