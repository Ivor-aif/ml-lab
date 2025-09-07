# 数据管理规范

## 概述

数据是机器学习项目的核心资产。本文档定义了ML Lab项目中数据的组织、存储、处理和管理规范，旨在确保数据的质量、安全性和可追溯性。

## 数据目录结构

```
data/
├── raw/                      # 原始数据（只读）
│   ├── dataset_name/         # 按数据集名称组织
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── metadata.json     # 数据集元信息
│   └── external/             # 外部数据源
│       ├── api_data/
│       └── web_scraped/
├── interim/                  # 中间处理数据
│   ├── cleaned/              # 清洗后的数据
│   ├── transformed/          # 转换后的数据
│   └── features/             # 特征工程结果
├── processed/                # 最终处理数据
│   ├── train/                # 训练数据
│   ├── validation/           # 验证数据
│   ├── test/                 # 测试数据
│   └── features/             # 特征数据
└── external/                 # 外部数据引用
    ├── references/           # 参考数据
    └── lookups/              # 查找表
```

## 数据命名规范

### 1. 文件命名

```
# 格式: [date]_[project]_[description]_[version].[extension]
# 示例:
2024-01-15_customer_segmentation_raw_v1.csv
2024-01-15_customer_segmentation_cleaned_v2.parquet
2024-01-15_customer_segmentation_features_v1.pkl
```

### 2. 数据集命名

```
# 格式: [domain]_[task]_[source]_[timeframe]
# 示例:
ecommerce_recommendation_clickstream_2024q1
finance_fraud_detection_transactions_2023
healthcare_diagnosis_imaging_2024
```

### 3. 变量命名

```python
# 使用小写字母和下划线
customer_id          # 客户ID
purchase_amount      # 购买金额
last_login_date      # 最后登录日期
is_premium_user      # 是否为高级用户

# 避免使用
CustomerID           # 驼峰命名
customer-id          # 连字符
cid                  # 缩写不明确
```

## 数据格式标准

### 1. 推荐格式

| 数据类型 | 推荐格式 | 使用场景 |
|---------|---------|----------|
| 结构化数据 | Parquet | 大数据集，需要压缩和快速读取 |
| 结构化数据 | CSV | 小数据集，需要人工查看 |
| 时间序列 | HDF5 | 高频时间序列数据 |
| 图像数据 | PNG/JPG | 原始图像 |
| 文本数据 | JSON/JSONL | 非结构化文本 |
| 模型特征 | Pickle | Python对象序列化 |

### 2. 数据类型规范

```python
# 推荐的pandas数据类型
data_types = {
    'customer_id': 'int64',           # 整数ID
    'email': 'string',                # 字符串
    'age': 'int8',                    # 小整数
    'income': 'float32',              # 浮点数
    'is_active': 'bool',              # 布尔值
    'signup_date': 'datetime64[ns]',  # 日期时间
    'category': 'category'            # 分类变量
}

# 应用数据类型
df = df.astype(data_types)
```

## 数据质量管理

### 1. 数据验证规则

```python
# 数据验证示例
def validate_dataset(df, schema):
    """
    数据集验证函数
    
    Args:
        df: pandas DataFrame
        schema: 数据模式定义
    
    Returns:
        validation_report: 验证报告
    """
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'errors': []
    }
    
    # 检查必需列
    required_columns = schema.get('required_columns', [])
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        validation_report['errors'].append(f"缺少必需列: {missing_columns}")
    
    # 检查数据范围
    for column, rules in schema.get('column_rules', {}).items():
        if column in df.columns:
            if 'min_value' in rules:
                invalid_count = (df[column] < rules['min_value']).sum()
                if invalid_count > 0:
                    validation_report['errors'].append(
                        f"{column}列有{invalid_count}个值小于最小值{rules['min_value']}"
                    )
    
    return validation_report

# 使用示例
schema = {
    'required_columns': ['customer_id', 'purchase_amount', 'purchase_date'],
    'column_rules': {
        'purchase_amount': {'min_value': 0, 'max_value': 10000},
        'age': {'min_value': 0, 'max_value': 120}
    }
}

report = validate_dataset(df, schema)
```

### 2. 数据清洗流程

```python
def clean_dataset(df):
    """
    标准数据清洗流程
    """
    # 1. 删除完全重复的行
    df = df.drop_duplicates()
    
    # 2. 处理缺失值
    # 数值型：使用中位数填充
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # 分类型：使用众数填充
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # 3. 处理异常值（使用IQR方法）
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 4. 标准化文本数据
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].str.strip().str.lower()
    
    return df
```

### 3. 数据质量监控

```python
def generate_data_quality_report(df, output_path):
    """
    生成数据质量报告
    """
    report = {
        'dataset_info': {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'creation_date': datetime.now().isoformat()
        },
        'column_analysis': {},
        'data_quality_score': 0
    }
    
    for column in df.columns:
        col_analysis = {
            'dtype': str(df[column].dtype),
            'non_null_count': df[column].count(),
            'null_count': df[column].isnull().sum(),
            'null_percentage': df[column].isnull().mean() * 100,
            'unique_count': df[column].nunique(),
            'unique_percentage': df[column].nunique() / len(df) * 100
        }
        
        if df[column].dtype in ['int64', 'float64']:
            col_analysis.update({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median()
            })
        
        report['column_analysis'][column] = col_analysis
    
    # 计算数据质量分数
    total_null_percentage = df.isnull().mean().mean() * 100
    quality_score = max(0, 100 - total_null_percentage)
    report['data_quality_score'] = quality_score
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    return report
```

## 数据版本控制

### 1. 数据版本管理

```python
import hashlib
import json
from datetime import datetime

def create_data_version(data_path, metadata):
    """
    创建数据版本记录
    """
    # 计算数据文件哈希值
    with open(data_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    version_info = {
        'version_id': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'file_path': data_path,
        'file_hash': file_hash,
        'file_size': os.path.getsize(data_path),
        'created_at': datetime.now().isoformat(),
        'metadata': metadata
    }
    
    # 保存版本信息
    version_file = data_path.replace('.csv', '_version.json')
    with open(version_file, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    return version_info

# 使用示例
metadata = {
    'description': '客户购买行为数据集',
    'source': 'CRM系统导出',
    'processing_steps': ['去重', '缺失值填充', '异常值处理'],
    'quality_score': 95.5
}

version_info = create_data_version('data/processed/customer_data.csv', metadata)
```

### 2. 数据血缘追踪

```python
class DataLineage:
    def __init__(self):
        self.lineage_graph = {}
    
    def add_transformation(self, input_data, output_data, transformation_info):
        """
        添加数据转换记录
        """
        if output_data not in self.lineage_graph:
            self.lineage_graph[output_data] = {
                'inputs': [],
                'transformations': []
            }
        
        self.lineage_graph[output_data]['inputs'].append(input_data)
        self.lineage_graph[output_data]['transformations'].append({
            'timestamp': datetime.now().isoformat(),
            'transformation': transformation_info
        })
    
    def get_lineage(self, data_name):
        """
        获取数据血缘信息
        """
        return self.lineage_graph.get(data_name, {})
    
    def export_lineage(self, output_path):
        """
        导出血缘图
        """
        with open(output_path, 'w') as f:
            json.dump(self.lineage_graph, f, indent=2)

# 使用示例
lineage = DataLineage()
lineage.add_transformation(
    input_data='raw_customer_data.csv',
    output_data='cleaned_customer_data.csv',
    transformation_info={
        'operation': 'data_cleaning',
        'script': 'src/data/clean_customer_data.py',
        'parameters': {'remove_duplicates': True, 'fill_missing': 'median'}
    }
)
```

## 数据安全和隐私

### 1. 敏感数据处理

```python
import hashlib
from cryptography.fernet import Fernet

def anonymize_pii(df, pii_columns):
    """
    匿名化个人身份信息
    """
    df_anonymized = df.copy()
    
    for column in pii_columns:
        if column in df.columns:
            # 使用哈希函数匿名化
            df_anonymized[column] = df[column].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:10]
            )
    
    return df_anonymized

def encrypt_sensitive_data(data, key):
    """
    加密敏感数据
    """
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data

# 使用示例
pii_columns = ['email', 'phone', 'ssn']
df_anonymized = anonymize_pii(df, pii_columns)
```

### 2. 数据访问控制

```python
class DataAccessManager:
    def __init__(self):
        self.access_log = []
        self.permissions = {}
    
    def grant_access(self, user, dataset, permission_level):
        """
        授予数据访问权限
        """
        if user not in self.permissions:
            self.permissions[user] = {}
        self.permissions[user][dataset] = permission_level
    
    def check_access(self, user, dataset, operation):
        """
        检查访问权限
        """
        user_perms = self.permissions.get(user, {})
        dataset_perm = user_perms.get(dataset, 'none')
        
        access_levels = {
            'read': ['read', 'write', 'admin'],
            'write': ['write', 'admin'],
            'delete': ['admin']
        }
        
        allowed = dataset_perm in access_levels.get(operation, [])
        
        # 记录访问日志
        self.access_log.append({
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'dataset': dataset,
            'operation': operation,
            'allowed': allowed
        })
        
        return allowed
```

## 数据备份和恢复

### 1. 自动备份策略

```python
import shutil
import schedule
import time

def backup_data(source_dir, backup_dir):
    """
    数据备份函数
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
    
    try:
        shutil.copytree(source_dir, backup_path)
        print(f"备份完成: {backup_path}")
        
        # 清理旧备份（保留最近7天）
        cleanup_old_backups(backup_dir, days=7)
        
    except Exception as e:
        print(f"备份失败: {e}")

def cleanup_old_backups(backup_dir, days=7):
    """
    清理旧备份文件
    """
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    for backup_folder in os.listdir(backup_dir):
        backup_path = os.path.join(backup_dir, backup_folder)
        if os.path.getctime(backup_path) < cutoff_time:
            shutil.rmtree(backup_path)
            print(f"删除旧备份: {backup_path}")

# 设置定时备份
schedule.every().day.at("02:00").do(
    backup_data, 
    source_dir='data/', 
    backup_dir='backups/'
)
```

### 2. 数据恢复流程

```bash
#!/bin/bash
# 数据恢复脚本 restore_data.sh

BACKUP_DIR="backups"
DATA_DIR="data"
RESTORE_DATE=$1

if [ -z "$RESTORE_DATE" ]; then
    echo "使用方法: ./restore_data.sh YYYYMMDD_HHMMSS"
    exit 1
fi

BACKUP_PATH="$BACKUP_DIR/backup_$RESTORE_DATE"

if [ ! -d "$BACKUP_PATH" ]; then
    echo "错误: 备份不存在 $BACKUP_PATH"
    exit 1
fi

# 备份当前数据
echo "备份当前数据..."
cp -r "$DATA_DIR" "${DATA_DIR}_backup_$(date +%Y%m%d_%H%M%S)"

# 恢复数据
echo "恢复数据从 $BACKUP_PATH..."
rm -rf "$DATA_DIR"
cp -r "$BACKUP_PATH" "$DATA_DIR"

echo "数据恢复完成"
```

## 性能优化

### 1. 大数据集处理

```python
import dask.dataframe as dd
from dask.distributed import Client

def process_large_dataset(file_path, chunk_size=10000):
    """
    分块处理大数据集
    """
    # 使用Dask处理大数据
    df = dd.read_csv(file_path)
    
    # 并行处理
    result = df.groupby('category').amount.sum().compute()
    
    return result

# 使用pandas分块读取
def process_in_chunks(file_path, chunk_size=10000):
    """
    分块处理数据
    """
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # 处理每个块
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    # 合并结果
    final_result = pd.concat(results, ignore_index=True)
    return final_result
```

### 2. 数据缓存策略

```python
import pickle
import os
from functools import wraps

def cache_data(cache_dir='cache'):
    """
    数据缓存装饰器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # 检查缓存是否存在
            if os.path.exists(cache_path):
                print(f"从缓存加载: {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            print(f"结果已缓存: {cache_path}")
            return result
        
        return wrapper
    return decorator

# 使用示例
@cache_data(cache_dir='data/cache')
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    # 复杂的数据处理逻辑
    processed_df = complex_processing(df)
    return processed_df
```

## 监控和告警

### 1. 数据质量监控

```python
class DataQualityMonitor:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.alerts = []
    
    def check_data_quality(self, df, dataset_name):
        """
        检查数据质量并生成告警
        """
        issues = []
        
        # 检查缺失值比例
        missing_ratio = df.isnull().mean().mean()
        if missing_ratio > self.thresholds.get('max_missing_ratio', 0.1):
            issues.append(f"缺失值比例过高: {missing_ratio:.2%}")
        
        # 检查重复行
        duplicate_ratio = df.duplicated().mean()
        if duplicate_ratio > self.thresholds.get('max_duplicate_ratio', 0.05):
            issues.append(f"重复行比例过高: {duplicate_ratio:.2%}")
        
        # 检查数据量变化
        expected_rows = self.thresholds.get('expected_rows', 0)
        if expected_rows > 0:
            row_change = abs(len(df) - expected_rows) / expected_rows
            if row_change > self.thresholds.get('max_row_change', 0.2):
                issues.append(f"数据量变化异常: {row_change:.2%}")
        
        if issues:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset_name,
                'issues': issues,
                'severity': 'high' if len(issues) > 2 else 'medium'
            }
            self.alerts.append(alert)
            self.send_alert(alert)
        
        return len(issues) == 0
    
    def send_alert(self, alert):
        """
        发送告警通知
        """
        print(f"数据质量告警: {alert}")
        # 这里可以集成邮件、Slack等通知方式
```

## 最佳实践总结

### 1. 数据组织原则
- 原始数据永远只读，不要修改
- 使用清晰的目录结构和命名规范
- 为每个数据集创建元数据文档
- 实施版本控制和血缘追踪

### 2. 数据质量保证
- 建立数据验证规则和质量检查
- 实施自动化的数据清洗流程
- 定期监控数据质量指标
- 记录所有数据处理步骤

### 3. 安全和合规
- 对敏感数据进行匿名化处理
- 实施访问控制和审计日志
- 定期备份重要数据
- 遵循数据保护法规

### 4. 性能优化
- 选择合适的数据格式
- 使用分块处理大数据集
- 实施缓存策略
- 监控存储和计算资源使用

---

**注意**: 数据管理是一个持续的过程，需要根据项目需求和数据特点不断优化和改进。请定期审查和更新数据管理流程。