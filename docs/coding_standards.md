# 代码规范和最佳实践

## 概述

本文档定义了ML Lab项目的代码规范和最佳实践，旨在确保代码的可读性、可维护性和团队协作的一致性。所有项目贡献者都应遵循这些规范。

## Python代码规范

### 1. 代码风格

#### 1.1 基本原则
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用 4 个空格进行缩进，不使用制表符
- 每行代码长度不超过 88 个字符（Black 格式化器标准）
- 使用有意义的变量名和函数名

#### 1.2 命名规范

```python
# 变量和函数：小写字母 + 下划线
user_count = 100
max_iterations = 1000

def calculate_accuracy(predictions, labels):
    pass

def preprocess_data(raw_data):
    pass

# 常量：大写字母 + 下划线
MAX_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'models/'

# 类名：驼峰命名法
class DataProcessor:
    pass

class RandomForestModel:
    pass

# 私有变量和方法：前缀下划线
class ModelTrainer:
    def __init__(self):
        self._model = None
        self.__private_key = "secret"
    
    def _prepare_data(self, data):
        pass
```

#### 1.3 导入规范

```python
# 标准库导入
import os
import sys
from datetime import datetime
from pathlib import Path

# 第三方库导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 本地模块导入
from src.data.preprocessing import clean_data
from src.models.base_model import BaseModel
from src.utils.logging import get_logger
```

### 2. 文档字符串规范

#### 2.1 函数文档

```python
def train_model(X_train, y_train, model_params=None, validation_split=0.2):
    """
    训练机器学习模型
    
    Args:
        X_train (pd.DataFrame): 训练特征数据
        y_train (pd.Series): 训练标签数据
        model_params (dict, optional): 模型参数字典. Defaults to None.
        validation_split (float, optional): 验证集比例. Defaults to 0.2.
    
    Returns:
        tuple: 包含训练好的模型和验证分数的元组
            - model: 训练好的模型对象
            - validation_score (float): 验证集上的分数
    
    Raises:
        ValueError: 当输入数据为空时抛出
        TypeError: 当模型参数类型不正确时抛出
    
    Example:
        >>> X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y_train = pd.Series([0, 1, 0])
        >>> model, score = train_model(X_train, y_train)
        >>> print(f"Validation score: {score:.3f}")
    """
    if X_train.empty or y_train.empty:
        raise ValueError("训练数据不能为空")
    
    # 实现代码...
    pass
```

#### 2.2 类文档

```python
class DataPreprocessor:
    """
    数据预处理器类
    
    该类提供了一套完整的数据预处理功能，包括数据清洗、特征工程、
    数据标准化等操作。
    
    Attributes:
        scaler (StandardScaler): 数据标准化器
        encoder (LabelEncoder): 标签编码器
        is_fitted (bool): 是否已经拟合数据
    
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> X_processed = preprocessor.fit_transform(X_raw)
        >>> X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(self, scaling_method='standard'):
        """
        初始化数据预处理器
        
        Args:
            scaling_method (str): 标准化方法，可选 'standard', 'minmax', 'robust'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.encoder = None
        self.is_fitted = False
```

### 3. 错误处理

#### 3.1 异常处理规范

```python
import logging
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

def load_dataset(file_path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """
    安全地加载数据集
    
    Args:
        file_path: 数据文件路径
        encoding: 文件编码格式
    
    Returns:
        加载的DataFrame，如果失败返回None
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        logger.info(f"成功加载数据集: {file_path}, 形状: {df.shape}")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"数据文件为空: {file_path}")
        return None
    except Exception as e:
        logger.error(f"加载数据集时发生未知错误: {e}")
        return None

def validate_model_input(X: pd.DataFrame, y: pd.Series) -> None:
    """
    验证模型输入数据
    
    Args:
        X: 特征数据
        y: 标签数据
    
    Raises:
        ValueError: 当数据验证失败时
    """
    if X.empty:
        raise ValueError("特征数据不能为空")
    
    if y.empty:
        raise ValueError("标签数据不能为空")
    
    if len(X) != len(y):
        raise ValueError(f"特征和标签数量不匹配: {len(X)} vs {len(y)}")
    
    if X.isnull().any().any():
        raise ValueError("特征数据包含缺失值")
    
    if y.isnull().any():
        raise ValueError("标签数据包含缺失值")
```

#### 3.2 自定义异常

```python
class MLLabException(Exception):
    """ML Lab项目基础异常类"""
    pass

class DataValidationError(MLLabException):
    """数据验证错误"""
    pass

class ModelTrainingError(MLLabException):
    """模型训练错误"""
    pass

class ConfigurationError(MLLabException):
    """配置错误"""
    pass

# 使用示例
def train_model_with_validation(X, y, config):
    try:
        validate_model_input(X, y)
        validate_config(config)
        
        model = create_model(config)
        model.fit(X, y)
        
        return model
        
    except DataValidationError as e:
        logger.error(f"数据验证失败: {e}")
        raise
    except ConfigurationError as e:
        logger.error(f"配置错误: {e}")
        raise
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise ModelTrainingError(f"训练过程中发生错误: {e}")
```

### 4. 类型注解

```python
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

def preprocess_features(
    data: pd.DataFrame,
    categorical_columns: List[str],
    numerical_columns: List[str],
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    预处理特征数据
    
    Args:
        data: 原始数据
        categorical_columns: 分类特征列名列表
        numerical_columns: 数值特征列名列表
        target_column: 目标列名（可选）
    
    Returns:
        处理后的特征数据和目标数据的元组
    """
    # 实现代码...
    pass

class ModelConfig:
    """模型配置类"""
    
    def __init__(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any],
        random_state: int = 42
    ) -> None:
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.random_state = random_state
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'random_state': self.random_state
        }
```

## 项目结构规范

### 1. 目录组织

```
src/
├── __init__.py
├── data/                     # 数据处理模块
│   ├── __init__.py
│   ├── preprocessing.py      # 数据预处理
│   ├── validation.py         # 数据验证
│   └── loaders.py           # 数据加载器
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── base_model.py        # 基础模型类
│   ├── sklearn_models.py    # Scikit-learn模型
│   ├── deep_learning.py     # 深度学习模型
│   └── ensemble.py          # 集成模型
├── training/                 # 训练相关
│   ├── __init__.py
│   ├── trainer.py           # 训练器
│   ├── hyperparameter_tuning.py  # 超参数调优
│   └── cross_validation.py  # 交叉验证
├── evaluation/               # 评估工具
│   ├── __init__.py
│   ├── metrics.py           # 评估指标
│   ├── visualization.py     # 结果可视化
│   └── reports.py           # 报告生成
└── utils/                    # 工具函数
    ├── __init__.py
    ├── logging.py           # 日志配置
    ├── config.py            # 配置管理
    ├── file_utils.py        # 文件操作
    └── decorators.py        # 装饰器
```

### 2. 模块设计原则

#### 2.1 基础模型类

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import joblib

class BaseModel(ABC):
    """
    所有模型的基础抽象类
    
    定义了模型的基本接口，所有具体模型都应该继承此类
    """
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.model_params = kwargs
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
        
        Returns:
            训练后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 预测特征
        
        Returns:
            预测结果
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            加载的模型实例
        """
        model_data = joblib.load(filepath)
        
        instance = cls(**model_data['model_params'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
```

#### 2.2 配置管理

```python
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """
    配置管理类
    
    负责加载和管理项目配置
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise ConfigurationError(f"配置文件不存在: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"配置文件格式错误: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            key: 配置键
            value: 新值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """
        保存配置到文件
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

# 全局配置实例
config = Config()
```

### 3. 日志配置

```python
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(
    log_level: str = 'INFO',
    log_file: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    设置项目日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
    
    Returns:
        配置好的logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建logger
    logger = logging.getLogger('mllab')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称
    
    Returns:
        logger实例
    """
    if name:
        return logging.getLogger(f'mllab.{name}')
    else:
        return logging.getLogger('mllab')

# 初始化日志
logger = setup_logging(
    log_level='INFO',
    log_file=f'logs/mllab_{datetime.now().strftime("%Y%m%d")}.log'
)
```

## 测试规范

### 1. 单元测试

```python
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.preprocessing import DataPreprocessor
from src.models.sklearn_models import RandomForestModel

class TestDataPreprocessor(unittest.TestCase):
    """
    数据预处理器测试类
    """
    
    def setUp(self):
        """测试前准备"""
        self.preprocessor = DataPreprocessor()
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
    
    def test_fit_transform(self):
        """测试fit_transform方法"""
        result = self.preprocessor.fit_transform(self.sample_data)
        
        # 检查返回类型
        self.assertIsInstance(result, pd.DataFrame)
        
        # 检查形状
        self.assertEqual(result.shape[0], self.sample_data.shape[0])
        
        # 检查是否已拟合
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_transform_without_fit(self):
        """测试未拟合时调用transform"""
        with self.assertRaises(ValueError):
            self.preprocessor.transform(self.sample_data)
    
    def test_empty_data(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.preprocessor.fit_transform(empty_data)
    
    @patch('src.data.preprocessing.logger')
    def test_logging(self, mock_logger):
        """测试日志记录"""
        self.preprocessor.fit_transform(self.sample_data)
        
        # 验证日志是否被调用
        mock_logger.info.assert_called()

class TestRandomForestModel(unittest.TestCase):
    """
    随机森林模型测试类
    """
    
    def setUp(self):
        """测试前准备"""
        self.model = RandomForestModel(n_estimators=10, random_state=42)
        
        # 创建测试数据
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        
        self.X_test = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20)
        })
    
    def test_fit_predict(self):
        """测试训练和预测"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 检查是否已拟合
        self.assertTrue(self.model.is_fitted)
        
        # 预测
        predictions = self.model.predict(self.X_test)
        
        # 检查预测结果
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_predict_without_fit(self):
        """测试未训练时预测"""
        with self.assertRaises(ValueError):
            self.model.predict(self.X_test)
    
    def test_save_load_model(self):
        """测试模型保存和加载"""
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 保存模型
        model_path = 'test_model.pkl'
        self.model.save_model(model_path)
        
        # 加载模型
        loaded_model = RandomForestModel.load_model(model_path)
        
        # 验证加载的模型
        self.assertTrue(loaded_model.is_fitted)
        
        # 比较预测结果
        original_pred = self.model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # 清理测试文件
        import os
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
```

### 2. 集成测试

```python
import pytest
import tempfile
import shutil
from pathlib import Path

from src.training.trainer import ModelTrainer
from src.data.loaders import load_dataset
from src.evaluation.metrics import calculate_metrics

class TestMLPipeline:
    """
    机器学习流水线集成测试
    """
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """创建示例数据集"""
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        data_path = Path(temp_dir) / 'sample_data.csv'
        data.to_csv(data_path, index=False)
        
        return str(data_path)
    
    def test_complete_pipeline(self, sample_dataset, temp_dir):
        """测试完整的ML流水线"""
        # 1. 加载数据
        data = load_dataset(sample_dataset)
        assert data is not None
        assert len(data) == 1000
        
        # 2. 分割数据
        X = data[['feature1', 'feature2']]
        y = data['target']
        
        # 3. 训练模型
        trainer = ModelTrainer(
            model_type='random_forest',
            model_params={'n_estimators': 10, 'random_state': 42}
        )
        
        model, metrics = trainer.train(X, y, test_size=0.2)
        
        # 4. 验证结果
        assert model is not None
        assert model.is_fitted
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        # 5. 保存模型
        model_path = Path(temp_dir) / 'trained_model.pkl'
        model.save_model(str(model_path))
        assert model_path.exists()
        
        # 6. 加载并验证模型
        loaded_model = type(model).load_model(str(model_path))
        assert loaded_model.is_fitted
        
        # 7. 预测测试
        test_X = X.sample(10)
        original_pred = model.predict(test_X)
        loaded_pred = loaded_model.predict(test_X)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
```

## 代码质量工具

### 1. 代码格式化

#### 1.1 Black配置 (pyproject.toml)

```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
# A regex preceded by ^/ will apply only to files and directories
# in the root of the project.
^/setup.py
'''
```

#### 1.2 isort配置

```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### 2. 代码检查

#### 2.1 flake8配置 (.flake8)

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, E266, E501, W503
max-complexity = 10
select = B,C,E,F,W,T4,B9
exclude = 
    .git,
    __pycache__,
    docs/source/conf.py,
    old,
    build,
    dist,
    .venv
```

#### 2.2 pylint配置 (.pylintrc)

```ini
[MASTER]
jobs=1
persistent=yes
safe-imports=no

[MESSAGES CONTROL]
disable=
    C0103,  # Invalid name
    C0111,  # Missing docstring
    R0903,  # Too few public methods
    R0913,  # Too many arguments
    W0613,  # Unused argument

[FORMAT]
max-line-length=88
indent-string='    '
```

### 3. 预提交钩子 (.pre-commit-config.yaml)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Git工作流规范

### 1. 分支命名规范

```bash
# 功能分支
feature/user-authentication
feature/data-preprocessing
feature/model-evaluation

# 修复分支
fix/data-loading-bug
fix/memory-leak-issue

# 热修复分支
hotfix/critical-security-patch

# 发布分支
release/v1.0.0
release/v1.1.0
```

### 2. 提交信息规范

```bash
# 格式: <type>(<scope>): <subject>

# 类型说明:
# feat: 新功能
# fix: 修复bug
# docs: 文档更新
# style: 代码格式调整
# refactor: 代码重构
# test: 测试相关
# chore: 构建过程或辅助工具的变动

# 示例:
feat(data): add data validation pipeline
fix(model): resolve memory leak in training loop
docs(readme): update installation instructions
refactor(utils): simplify logging configuration
test(preprocessing): add unit tests for data cleaning
```

### 3. Pull Request规范

```markdown
## 变更描述
简要描述本次变更的内容和目的

## 变更类型
- [ ] 新功能
- [ ] Bug修复
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 其他

## 测试
- [ ] 已添加单元测试
- [ ] 已添加集成测试
- [ ] 所有测试通过
- [ ] 手动测试通过

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 已更新相关文档
- [ ] 已添加必要的注释
- [ ] 无明显的代码质量问题

## 相关Issue
关闭 #123
```

## 性能优化指南

### 1. 代码性能

```python
# 使用生成器而不是列表推导（大数据集）
def process_large_dataset(data):
    # 好的做法
    return (process_item(item) for item in data)
    
    # 避免（内存消耗大）
    # return [process_item(item) for item in data]

# 使用适当的数据结构
def count_items(items):
    # 好的做法
    from collections import Counter
    return Counter(items)
    
    # 避免
    # counts = {}
    # for item in items:
    #     counts[item] = counts.get(item, 0) + 1
    # return counts

# 缓存昂贵的计算
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(n):
    # 昂贵的计算逻辑
    return result
```

### 2. 内存优化

```python
# 使用__slots__减少内存使用
class DataPoint:
    __slots__ = ['x', 'y', 'label']
    
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

# 及时释放大对象
def process_large_data():
    large_data = load_large_dataset()
    
    try:
        result = process(large_data)
    finally:
        del large_data  # 显式删除
    
    return result

# 使用生成器处理大文件
def read_large_file(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            yield line.strip()
```

## 安全编码规范

### 1. 输入验证

```python
def safe_file_operation(filepath: str) -> bool:
    """
    安全的文件操作
    """
    # 验证文件路径
    if not isinstance(filepath, str):
        raise TypeError("文件路径必须是字符串")
    
    # 防止路径遍历攻击
    if '..' in filepath or filepath.startswith('/'):
        raise ValueError("不安全的文件路径")
    
    # 检查文件扩展名
    allowed_extensions = {'.csv', '.json', '.parquet'}
    if not any(filepath.endswith(ext) for ext in allowed_extensions):
        raise ValueError("不支持的文件类型")
    
    return True

def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证模型参数
    """
    validated_params = {}
    
    # 验证学习率
    if 'learning_rate' in params:
        lr = params['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            raise ValueError("学习率必须在(0, 1]范围内")
        validated_params['learning_rate'] = lr
    
    # 验证批次大小
    if 'batch_size' in params:
        bs = params['batch_size']
        if not isinstance(bs, int) or bs <= 0 or bs > 10000:
            raise ValueError("批次大小必须在(0, 10000]范围内")
        validated_params['batch_size'] = bs
    
    return validated_params
```

### 2. 敏感信息处理

```python
import os
from typing import Optional

def get_api_key(key_name: str) -> Optional[str]:
    """
    安全地获取API密钥
    """
    # 从环境变量获取
    api_key = os.getenv(key_name)
    
    if not api_key:
        logger.warning(f"未找到API密钥: {key_name}")
        return None
    
    # 不要在日志中记录完整的密钥
    logger.info(f"已加载API密钥: {key_name[:4]}****")
    
    return api_key

# 避免在代码中硬编码敏感信息
class DatabaseConfig:
    def __init__(self):
        # 好的做法：从环境变量读取
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '5432'))
        self.username = os.getenv('DB_USERNAME')
        self.password = os.getenv('DB_PASSWORD')
        
        # 避免：硬编码敏感信息
        # self.password = "hardcoded_password"  # 不要这样做！
```

## 总结

遵循这些代码规范和最佳实践将帮助您：

1. **提高代码质量**: 编写更清晰、更可维护的代码
2. **增强团队协作**: 统一的代码风格便于团队成员理解和维护
3. **减少错误**: 通过测试和验证降低bug发生率
4. **提升性能**: 优化代码执行效率和资源使用
5. **保障安全**: 防范常见的安全漏洞和风险

请定期审查和更新这些规范，以适应项目发展和技术进步的需要。

---

**注意**: 这些规范是指导性的，可以根据具体项目需求进行调整。重要的是保持一致性和团队共识。