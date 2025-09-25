"""
Utility functions for Optimistic Estimation Experiments
"""

import os
import json
import pickle
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from config import RANDOM_SEED, RESULTS_DIR, PLOTS_DIR, VISUALIZATION, LOGGING

# Set random seeds for reproducibility
def set_random_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Directory management
def ensure_dir(directory: str) -> str:
    """Ensure directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_results_dir(experiment_name: str) -> str:
    """Get results directory for specific experiment."""
    return ensure_dir(os.path.join(RESULTS_DIR, experiment_name))

def get_plots_dir(experiment_name: str) -> str:
    """Get plots directory for specific experiment."""
    return ensure_dir(os.path.join(PLOTS_DIR, experiment_name))

# Data persistence
def save_results(results: Dict[str, Any], filename: str, experiment_name: str):
    """Save experiment results to JSON file."""
    results_dir = get_results_dir(experiment_name)
    filepath = os.path.join(results_dir, f"{filename}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logging.info(f"Results saved to {filepath}")

def load_results(filename: str, experiment_name: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    results_dir = get_results_dir(experiment_name)
    filepath = os.path.join(results_dir, f"{filename}.json")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def save_model(model, filename: str, experiment_name: str):
    """Save model using pickle."""
    results_dir = get_results_dir(experiment_name)
    filepath = os.path.join(results_dir, f"{filename}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Model saved to {filepath}")

def load_model(filename: str, experiment_name: str):
    """Load model using pickle."""
    results_dir = get_results_dir(experiment_name)
    filepath = os.path.join(results_dir, f"{filename}.pkl")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Logging setup
def setup_logging(experiment_name: str):
    """Setup logging configuration."""
    log_dir = ensure_dir(os.path.join(RESULTS_DIR, experiment_name))
    log_file = os.path.join(log_dir, LOGGING['file'])
    
    logging.basicConfig(
        level=getattr(logging, LOGGING['level']),
        format=LOGGING['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Visualization utilities
def setup_plotting():
    """Setup matplotlib and seaborn plotting parameters."""
    plt.style.use(VISUALIZATION['style'])
    sns.set_palette(VISUALIZATION['color_palette'])
    plt.rcParams['figure.figsize'] = VISUALIZATION['figure_size']
    plt.rcParams['figure.dpi'] = VISUALIZATION['dpi']
    plt.rcParams['font.size'] = VISUALIZATION['font_size']

def save_plot(fig, filename: str, experiment_name: str):
    """Save plot to file."""
    plots_dir = get_plots_dir(experiment_name)
    filepath = os.path.join(plots_dir, f"{filename}.{VISUALIZATION['save_format']}")
    fig.savefig(filepath, dpi=VISUALIZATION['dpi'], bbox_inches='tight')
    logging.info(f"Plot saved to {filepath}")

# Mathematical utilities
def frobenius_norm(matrix: np.ndarray) -> float:
    """Compute Frobenius norm of a matrix."""
    return np.sqrt(np.sum(matrix ** 2))

def relative_error(true_matrix: np.ndarray, estimated_matrix: np.ndarray) -> float:
    """Compute relative error between two matrices."""
    return frobenius_norm(true_matrix - estimated_matrix) / frobenius_norm(true_matrix)

def compute_condition_number(matrix: np.ndarray) -> float:
    """Compute condition number of a matrix."""
    return np.linalg.cond(matrix)

def compute_rank(matrix: np.ndarray, tolerance: float = 1e-10) -> int:
    """Compute numerical rank of a matrix."""
    s = np.linalg.svd(matrix, compute_uv=False)
    return np.sum(s > tolerance)

# Statistical utilities
def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for data."""
    from scipy import stats
    
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    
    return mean - h, mean + h

def bootstrap_confidence_interval(data: np.ndarray, statistic_func=np.mean, 
                                n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return lower, upper

# Matrix generation utilities
def generate_low_rank_matrix(shape: Tuple[int, int], rank: int, 
                           noise_level: float = 0.0) -> np.ndarray:
    """Generate a low-rank matrix with optional noise."""
    m, n = shape
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    matrix = U @ V
    
    if noise_level > 0:
        noise = np.random.randn(m, n) * noise_level
        matrix += noise
    
    return matrix

def generate_observation_mask(shape: Tuple[int, int], ratio: float, 
                            strategy: str = 'random') -> np.ndarray:
    """Generate observation mask for matrix completion."""
    m, n = shape
    total_entries = m * n
    num_observed = int(total_entries * ratio)
    
    mask = np.zeros((m, n), dtype=bool)
    
    if strategy == 'random':
        indices = np.random.choice(total_entries, num_observed, replace=False)
        mask.flat[indices] = True
    
    elif strategy == 'structured':
        # Observe complete rows and columns
        num_rows = int(np.sqrt(num_observed * m / n))
        num_cols = int(np.sqrt(num_observed * n / m))
        row_indices = np.random.choice(m, num_rows, replace=False)
        col_indices = np.random.choice(n, num_cols, replace=False)
        mask[row_indices, :] = True
        mask[:, col_indices] = True
    
    elif strategy == 'clustered':
        # Create clustered observations
        center_row = np.random.randint(0, m)
        center_col = np.random.randint(0, n)
        
        for _ in range(num_observed):
            # Sample around the center with decreasing probability
            row = np.clip(int(np.random.normal(center_row, m/4)), 0, m-1)
            col = np.clip(int(np.random.normal(center_col, n/4)), 0, n-1)
            mask[row, col] = True
    
    elif strategy == 'diagonal':
        # Focus on diagonal and nearby entries
        for i in range(min(m, n)):
            mask[i, i] = True
        
        remaining = num_observed - min(m, n)
        for _ in range(remaining):
            i = np.random.randint(0, m)
            j = np.random.randint(0, n)
            if abs(i - j) <= 2:  # Near diagonal
                mask[i, j] = True
    
    elif strategy == 'corners':
        # Focus on corners
        corner_size = int(np.sqrt(num_observed / 4))
        mask[:corner_size, :corner_size] = True
        mask[-corner_size:, :corner_size] = True
        mask[:corner_size, -corner_size:] = True
        mask[-corner_size:, -corner_size:] = True
    
    return mask

# Progress tracking
class ExperimentTracker:
    """Track experiment progress and results."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start tracking experiment."""
        import time
        self.start_time = time.time()
        logging.info(f"Starting experiment: {self.experiment_name}")
    
    def end(self):
        """End tracking experiment."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logging.info(f"Experiment {self.experiment_name} completed in {duration:.2f} seconds")
    
    def log_result(self, key: str, value: Any):
        """Log a result."""
        self.results[key] = value
        logging.info(f"{key}: {value}")
    
    def save_results(self, filename: str = None):
        """Save all tracked results."""
        if filename is None:
            filename = f"{self.experiment_name}_results"
        
        self.results['experiment_name'] = self.experiment_name
        self.results['start_time'] = self.start_time
        self.results['end_time'] = self.end_time
        self.results['duration'] = self.end_time - self.start_time if self.end_time else None
        
        save_results(self.results, filename, self.experiment_name)