# 频率特性实验使用指南

## 概述

本实验旨在研究神经网络对频率特征的学习能力。通过生成具有明显频率特征的数据样本，并使用两层神经网络进行拟合，观察模型对不同频率成分的学习效果。

## 环境配置

### 1. 安装依赖

```bash
# 安装Python依赖包
pip install -r requirements.txt

# 或使用conda
conda install pytorch torchvision numpy matplotlib seaborn scikit-learn pandas
```

### 2. 验证安装

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
print("环境配置成功！")
```

## 快速开始

### 1. 运行基础实验

```bash
# 运行完整的基础实验
python experiment.py --mode single

# 或直接运行训练脚本
python train.py
```

### 2. 生成数据

```bash
# 仅生成和可视化数据
python data_generator.py
```

### 3. 可视化结果

```bash
# 生成实验报告（需要先运行实验）
python experiment.py --mode report --experiment freq1_baseline
```

## 详细使用说明

### 配置参数

实验的所有参数都在 `config.py` 中定义。主要参数包括：

#### 数据生成参数

```python
# 频率函数参数
coefficients = {
    'a0': 2.0,      # 常数项
    'a1': 1.5,      # 1倍频幅度
    'b1': 0.0,      # 1倍频相位
    'a2': 0.8,      # 2倍频幅度
    'b2': np.pi/4,  # 2倍频相位
    'a3': 0.4,      # 3倍频幅度
    'b3': np.pi/2,  # 3倍频相位
}

# 采样参数
num_samples = 1000        # 训练样本数量
num_test_samples = 200    # 测试样本数量
x_range = (-2*np.pi, 2*np.pi)  # 采样范围
noise_level = 0.05        # 噪声标准差
```

#### 模型参数

```python
model_params = {
    'input_dim': 1,           # 输入维度
    'hidden_dim': 64,         # 隐藏层维度
    'output_dim': 1,          # 输出维度
    'activation': 'relu',     # 激活函数
    'dropout_rate': 0.1,      # Dropout率
}
```

#### 训练参数

```python
training_params = {
    'learning_rate': 0.001,   # 学习率
    'batch_size': 32,         # 批次大小
    'num_epochs': 500,        # 训练轮数
    'validation_split': 0.2,  # 验证集比例
}
```

### 修改实验配置

#### 方法1：直接修改config.py

```python
# 在config.py中修改默认参数
self.data_params = {
    'coefficients': {
        'a0': 3.0,      # 修改常数项
        'a1': 2.0,      # 修改1倍频幅度
        # ...
    }
}
```

#### 方法2：在代码中动态修改

```python
from config import ExperimentConfig

config = ExperimentConfig()
config.update_config({
    'model_params': {'hidden_dim': 128},
    'training_params': {'learning_rate': 0.01}
})
```

### 运行不同实验变体

#### 1. 单个实验变体

```bash
# 运行不同隐藏层大小的实验
python experiment.py --mode single --variant hidden_128

# 运行不同学习率的实验
python experiment.py --mode single --variant lr_0_0100

# 运行不同激活函数的实验
python experiment.py --mode single --variant activation_tanh
```

#### 2. 批量实验

```bash
# 运行多个实验变体并比较结果
python experiment.py --mode batch
```

### 分析实验结果

#### 1. 查看训练日志

实验结果保存在 `results/` 目录下：

```
results/
├── freq1_baseline/
│   ├── model.pth              # 训练好的模型
│   ├── training_history.json  # 训练历史
│   ├── experiment_config.json # 实验配置
│   ├── data/
│   │   └── experiment_data.npz # 实验数据
│   └── plots/                 # 可视化图像
│       ├── function_fitting.png
│       ├── training_history.png
│       ├── frequency_analysis.png
│       └── error_analysis.png
```

#### 2. 生成可视化报告

```bash
# 为特定实验生成可视化报告
python experiment.py --mode report --experiment freq1_baseline

# 或直接运行可视化脚本
python visualize.py
```

#### 3. 加载和分析模型

```python
from config import ExperimentConfig
from train import load_model
from data_generator import FrequencyDataGenerator

# 加载模型
config = ExperimentConfig()
model = load_model('results/freq1_baseline/model.pth', config)

# 生成测试数据
generator = FrequencyDataGenerator(config)
x_test = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

# 可视化预测结果
import matplotlib.pyplot as plt
plt.plot(x_test, y_pred, label='预测')
plt.show()
```

## 实验示例

### 示例1：基础频率函数拟合

```python
# 定义简单的频率函数：f(x) = 2 + sin(x) + 0.5*sin(2x)
from config import ExperimentConfig
from experiment import run_single_experiment

variant = {
    'name': 'simple_freq',
    'description': '简单频率函数',
    'config_updates': {
        'data_params': {
            'coefficients': {
                'a0': 2.0,
                'a1': 1.0,
                'b1': 0.0,
                'a2': 0.5,
                'b2': 0.0
            }
        }
    }
}

config = ExperimentConfig()
results = run_single_experiment(variant, config)
print(f"R² Score: {results['evaluation_results']['r2_score']:.4f}")
```

### 示例2：高噪声环境下的学习

```python
# 测试模型在高噪声环境下的表现
variant = {
    'name': 'high_noise',
    'description': '高噪声环境',
    'config_updates': {
        'data_params': {'noise_level': 0.2},
        'training_params': {'num_epochs': 1000}  # 增加训练轮数
    }
}

results = run_single_experiment(variant, config)
```

### 示例3：复杂频率成分

```python
# 测试复杂频率成分的学习能力
variant = {
    'name': 'complex_freq',
    'description': '复杂频率成分',
    'config_updates': {
        'data_params': {
            'coefficients': {
                'a0': 1.0,
                'a1': 1.5, 'b1': 0.0,
                'a2': 0.8, 'b2': np.pi/4,
                'a3': 0.4, 'b3': np.pi/2,
                'a4': 0.2, 'b4': np.pi,
                'a5': 0.1, 'b5': 3*np.pi/2
            }
        },
        'model_params': {'hidden_dim': 128}  # 增加模型容量
    }
}

results = run_single_experiment(variant, config)
```

## 评估指标说明

### 1. R² Score (决定系数)
- 范围：(-∞, 1]
- 1表示完美拟合，0表示模型性能等同于简单平均值
- 负值表示模型性能差于简单平均值

### 2. RMSE (均方根误差)
- 范围：[0, +∞)
- 值越小表示拟合效果越好
- 与目标变量具有相同的量纲

### 3. MAE (平均绝对误差)
- 范围：[0, +∞)
- 对异常值不如RMSE敏感
- 更直观地反映平均误差大小

### 4. MSE (均方误差)
- 范围：[0, +∞)
- 对大误差更敏感
- 常用作训练时的损失函数

## 常见问题

### Q1: 训练损失不收敛怎么办？

A: 尝试以下解决方案：
1. 降低学习率：`learning_rate = 0.0001`
2. 增加训练轮数：`num_epochs = 1000`
3. 调整网络结构：增加隐藏层维度
4. 检查数据范围：确保输入数据已正确归一化

### Q2: 模型过拟合怎么办？

A: 可以尝试：
1. 增加Dropout率：`dropout_rate = 0.3`
2. 减少模型复杂度：`hidden_dim = 32`
3. 增加训练数据：`num_samples = 2000`
4. 早停：`early_stopping_patience = 20`

### Q3: 如何处理复杂的频率成分？

A: 对于复杂频率：
1. 增加网络容量：`hidden_dim = 256`
2. 使用更好的激活函数：`activation = 'gelu'`
3. 调整学习率调度：使用余弦退火
4. 增加训练数据和训练轮数

### Q4: 可视化图像显示异常？

A: 检查以下设置：
1. 确保安装了中文字体支持
2. 检查matplotlib后端设置
3. 验证数据范围和格式
4. 确认保存路径存在

## 扩展实验

### 1. 不同网络架构
- 三层网络
- 残差连接
- 注意力机制

### 2. 不同优化算法
- SGD with momentum
- AdamW
- RMSprop

### 3. 正则化技术
- L1/L2正则化
- Batch Normalization
- Layer Normalization

### 4. 数据增强
- 添加不同类型的噪声
- 数据平移和缩放
- 频率域增强

## 参考资料

1. [PyTorch官方文档](https://pytorch.org/docs/)
2. [神经网络与深度学习](https://nndl.github.io/)
3. [频域分析基础](https://en.wikipedia.org/wiki/Frequency_domain)
4. [函数拟合理论](https://en.wikipedia.org/wiki/Curve_fitting)