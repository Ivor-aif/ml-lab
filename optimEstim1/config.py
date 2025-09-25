"""
Configuration file for Optimistic Estimation Experiments
"""

import numpy as np

# General experiment settings
RANDOM_SEED = 42
DEVICE = 'cpu'  # 'cuda' if available
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

# Matrix Factorization Experiment Configuration
MATRIX_FACTORIZATION = {
    'matrix_size': 5,
    'rank': 2,
    'init_scales': [1e-4, 1e-3, 1e-2, 1e-1],  # Small initialization scales
    'sample_sizes': list(range(5, 26, 2)),  # Sample sizes to test (max 25 for 5x5 matrix)
    'num_trials': 10,  # Number of random trials per configuration
    'max_iterations': 5000,
    'learning_rate': 0.01,
    'tolerance': 1e-6,
    'noise_level': 0.01,
}

# Matrix Completion Experiment Configuration
MATRIX_COMPLETION = {
    'matrix_size': (10, 10),
    'rank': 3,
    'completion_ratios': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'position_strategies': [
        'random',      # Random observation positions
        'structured',  # Structured patterns (rows/columns)
        'clustered',   # Clustered observations
        'diagonal',    # Diagonal patterns
        'corners',     # Corner-focused observations
    ],
    'num_trials': 20,
    'max_iterations': 3000,
    'learning_rate': 0.01,
    'tolerance': 1e-6,
    'init_scale': 1e-3,
}

# Neural Network Experiment Configuration
NEURAL_NETWORK = {
    'input_dim': 2,
    'output_dim': 1,
    'hidden_sizes': [32, 64, 128],
    'activation': 'relu',
    'target_functions': [
        'polynomial',   # Polynomial target functions
        'trigonometric', # Trigonometric functions
        'gaussian',     # Gaussian mixtures
        'piecewise',    # Piecewise linear functions
    ],
    'sample_sizes': list(range(10, 201, 10)),
    'num_trials': 15,
    'max_epochs': 2000,
    'learning_rate': 1e-3,  # Single learning rate for consistency
    'learning_rates': [1e-3, 1e-4],
    'batch_size': 32,
    'early_stopping_patience': 100,
    'tolerance': 1e-5,
    'target_error': 0.1,  # Target error threshold for optimistic bound verification
}

# Target Function Parameters
TARGET_FUNCTIONS = {
    'polynomial': {
        'degree': 3,
        'coefficients_range': (-2, 2),
        'noise_level': 0.05,
    },
    'trigonometric': {
        'num_components': 3,
        'frequency_range': (0.5, 3.0),
        'amplitude_range': (0.5, 2.0),
        'phase_range': (0, 2*np.pi),
        'noise_level': 0.05,
    },
    'gaussian': {
        'num_components': 2,
        'center_range': (-2, 2),
        'width_range': (0.3, 1.0),
        'amplitude_range': (0.5, 2.0),
        'noise_level': 0.05,
    },
    'piecewise': {
        'num_pieces': 4,
        'breakpoint_range': (-2, 2),
        'slope_range': (-2, 2),
        'noise_level': 0.05,
    }
}

# Visualization Configuration
VISUALIZATION = {
    'figure_size': (10, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'Set2',
    'font_size': 12,
    'save_format': 'png',
}

# Analysis Configuration
ANALYSIS = {
    'confidence_level': 0.95,
    'significance_level': 0.05,
    'bootstrap_samples': 1000,
    'statistical_tests': ['t_test', 'wilcoxon', 'ks_test'],
}

# Optimistic Estimation Theory Parameters
OPTIMISTIC_THEORY = {
    'linear_approximation_radius': 0.1,
    'condition_number_threshold': 100,
    'eigenvalue_threshold': 1e-8,
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'experiment.log',
}

# Performance Metrics
METRICS = {
    'matrix_factorization': [
        'frobenius_error',
        'relative_error',
        'convergence_rate',
        'sample_efficiency',
    ],
    'matrix_completion': [
        'completion_error',
        'observed_error',
        'unobserved_error',
        'position_sensitivity',
    ],
    'neural_network': [
        'train_loss',
        'test_loss',
        'generalization_gap',
        'sample_complexity',
    ],
}

# Export configuration for easy access
__all__ = [
    'RANDOM_SEED',
    'DEVICE',
    'RESULTS_DIR',
    'PLOTS_DIR',
    'MATRIX_FACTORIZATION',
    'MATRIX_COMPLETION',
    'NEURAL_NETWORK',
    'TARGET_FUNCTIONS',
    'VISUALIZATION',
    'ANALYSIS',
    'OPTIMISTIC_THEORY',
    'LOGGING',
    'METRICS',
]