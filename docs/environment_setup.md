# 环境配置指南

## 概述

本指南将帮助您设置ML Lab项目的开发环境。我们支持多种环境配置方式，您可以根据自己的需求选择最适合的方案。

## 系统要求

### 最低要求
- **操作系统**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 或更高版本
- **内存**: 8GB RAM (推荐 16GB+)
- **存储**: 至少 10GB 可用空间
- **GPU**: 可选，支持CUDA 11.0+ (用于深度学习加速)

### 推荐配置
- **CPU**: Intel i7/AMD Ryzen 7 或更高
- **内存**: 32GB RAM
- **GPU**: NVIDIA RTX 3070 或更高 (8GB+ VRAM)
- **存储**: SSD 硬盘

## 环境配置方案

### 方案一：Conda环境 (推荐)

#### 1. 安装Anaconda/Miniconda

**Windows:**
```powershell
# 下载并安装Miniconda
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"
.\miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
```

**macOS:**
```bash
# 使用Homebrew安装
brew install --cask miniconda

# 或直接下载安装
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### 2. 创建Conda环境

```bash
# 克隆项目
git clone <your-repository-url>
cd mlLab

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate mllab

# 验证安装
python --version
pip list
```

#### 3. environment.yml 配置文件

```yaml
name: mllab
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - numpy=1.21.0
  - pandas=1.3.0
  - matplotlib=3.4.0
  - seaborn=0.11.0
  - scikit-learn=1.0.0
  - jupyter=1.0.0
  - jupyterlab=3.1.0
  - pytorch=1.9.0
  - torchvision=0.10.0
  - cudatoolkit=11.1
  - pip=21.0
  - pip:
    - tensorflow==2.6.0
    - xgboost==1.4.0
    - lightgbm==3.2.0
    - optuna==2.9.0
    - mlflow==1.20.0
    - wandb==0.12.0
    - plotly==5.3.0
    - streamlit==0.86.0
```

### 方案二：Python虚拟环境

#### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv mllab_env

# 激活环境
# Windows
mllab_env\Scripts\activate
# macOS/Linux
source mllab_env/bin/activate

# 升级pip
pip install --upgrade pip
```

#### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

#### 3. requirements.txt 配置文件

```txt
# 核心数据科学库
numpy==1.21.0
pandas==1.3.0
scipy==1.7.0
matplotlib==3.4.0
seaborn==0.11.0
plotly==5.3.0

# 机器学习库
scikit-learn==1.0.0
xgboost==1.4.0
lightgbm==3.2.0

# 深度学习框架
tensorflow==2.6.0
torch==1.9.0
torchvision==0.10.0

# Jupyter环境
jupyter==1.0.0
jupyterlab==3.1.0
ipykernel==6.0.0

# 实验跟踪和优化
mlflow==1.20.0
wandb==0.12.0
optuna==2.9.0

# 数据处理和可视化
streamlit==0.86.0
dash==1.21.0

# 工具库
tqdm==4.62.0
click==8.0.0
pyyaml==5.4.0
joblib==1.0.0
```

### 方案三：Docker环境

#### 1. Dockerfile

```dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8888 8501

# 启动命令
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

#### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  mllab:
    build: .
    ports:
      - "8888:8888"  # Jupyter Lab
      - "8501:8501"  # Streamlit
      - "5000:5000"  # MLflow
    volumes:
      - .:/app
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
    environment:
      - JUPYTER_ENABLE_LAB=yes
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  mlflow:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000
```

#### 3. 使用Docker

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f mllab

# 进入容器
docker-compose exec mllab bash

# 停止服务
docker-compose down
```

## GPU支持配置

### NVIDIA GPU (CUDA)

#### 1. 安装NVIDIA驱动

```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-470

# 重启系统
sudo reboot

# 验证安装
nvidia-smi
```

#### 2. 安装CUDA Toolkit

```bash
# 下载并安装CUDA 11.1
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run

# 设置环境变量
echo 'export PATH=/usr/local/cuda-11.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 3. 验证GPU支持

```python
# TensorFlow GPU测试
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# PyTorch GPU测试
import torch
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device Count: ", torch.cuda.device_count())
print("Current Device: ", torch.cuda.current_device())
```

## 开发工具配置

### 1. Jupyter Lab扩展

```bash
# 安装常用扩展
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install plotlywidget
jupyter labextension install @jupyterlab/toc
jupyter labextension install @ryantam626/jupyterlab_code_formatter

# 启用扩展
jupyter lab build
```

### 2. VS Code配置

安装推荐扩展：
- Python
- Jupyter
- Python Docstring Generator
- GitLens
- Black Formatter
- Pylance

#### settings.json配置

```json
{
    "python.defaultInterpreterPath": "./mllab_env/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "jupyter.askForKernelRestart": false,
    "files.autoSave": "afterDelay",
    "editor.formatOnSave": true
}
```

### 3. Git配置

```bash
# 配置Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置.gitignore
echo "# ML Lab .gitignore" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "data/raw/" >> .gitignore
echo "models/*.pkl" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo "mlruns/" >> .gitignore
```

## 环境验证

### 验证脚本

创建 `scripts/verify_environment.py`：

```python
#!/usr/bin/env python3
"""
环境验证脚本
"""

import sys
import importlib
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        return False

def check_packages():
    """检查必要的包"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
        'jupyter', 'tensorflow', 'torch', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_gpu_support():
    """检查GPU支持"""
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        print(f"✓ TensorFlow GPU支持: {gpu_available}")
    except:
        print("✗ TensorFlow GPU检查失败")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✓ PyTorch CUDA支持: {cuda_available}")
    except:
        print("✗ PyTorch CUDA检查失败")

def main():
    print("ML Lab 环境验证")
    print("=" * 30)
    
    python_ok = check_python_version()
    packages_ok = check_packages()
    
    print("\nGPU支持检查:")
    check_gpu_support()
    
    if python_ok and packages_ok:
        print("\n✓ 环境配置完成！")
        return 0
    else:
        print("\n✗ 环境配置存在问题，请检查上述错误")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### 运行验证

```bash
# 运行环境验证
python scripts/verify_environment.py

# 或使用make命令（如果有Makefile）
make verify
```

## 常见问题解决

### Q1: Conda环境创建失败

```bash
# 清理Conda缓存
conda clean --all

# 更新Conda
conda update conda

# 重新创建环境
conda env create -f environment.yml --force
```

### Q2: CUDA版本不匹配

```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 安装对应版本的PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Q3: Jupyter内核问题

```bash
# 安装内核
python -m ipykernel install --user --name mllab --display-name "ML Lab"

# 刷新内核列表
jupyter kernelspec list
```

### Q4: 包版本冲突

```bash
# 使用pip-tools管理依赖
pip install pip-tools

# 生成锁定版本文件
pip-compile requirements.in

# 安装锁定版本
pip-sync requirements.txt
```

## 性能优化建议

1. **使用SSD存储**: 提高数据读取速度
2. **增加内存**: 减少磁盘交换，提高处理大数据集的能力
3. **GPU加速**: 用于深度学习模型训练
4. **并行处理**: 利用多核CPU进行数据处理
5. **缓存机制**: 避免重复计算

## 更新和维护

### 定期更新

```bash
# 更新Conda环境
conda env update -f environment.yml

# 更新pip包
pip install --upgrade -r requirements.txt

# 更新Jupyter扩展
jupyter lab build
```

### 环境备份

```bash
# 导出Conda环境
conda env export > environment_backup.yml

# 导出pip依赖
pip freeze > requirements_backup.txt
```

---

**注意**: 环境配置可能因系统差异而有所不同。如遇到问题，请参考官方文档或提交Issue寻求帮助。