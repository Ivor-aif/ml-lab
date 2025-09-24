"""
Neural network condensation phenomenon experiment configuration file
"""
import argparse
import torch

def get_config():
    """Get experiment configuration parameters"""
    parser = argparse.ArgumentParser(description='Neural network condensation phenomenon experiment')
    
    # Basic training parameters
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer: sgd | adam')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    
    # Data parameters
    parser.add_argument('--training_size', default=100, type=int, help='Training set size')
    parser.add_argument('--test_size', default=1000, type=int, help='Test set size')
    parser.add_argument('--boundary', nargs='+', type=str, default=['-1', '1'], help='1D data boundary')
    
    # Network structure parameters
    parser.add_argument('--input_dim', default=1, type=int, help='Input dimension')
    parser.add_argument('--output_dim', default=1, type=int, help='Output dimension')
    parser.add_argument('--hidden_layers_width', nargs='+', type=int, default=[100], help='Hidden layer width')
    parser.add_argument('--gamma', type=float, default=1.0, help='Parameter initialization distribution variance power')
    
    # Activation function parameters
    parser.add_argument('--act_func_name', default='ReLU', help='Activation function name')
    
    # Experiment recording parameters
    parser.add_argument('--save_epoch', default=500, type=int, help='Save interval')
    parser.add_argument('--plot_epoch', default=500, type=int, help='Plot interval')
    parser.add_argument('--rand_seed', default=42, type=int, help='Random seed')
    
    # Device parameters
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                       type=str, help='Training device')
    
    args, _ = parser.parse_known_args()
    return args

# Activation function configuration
ACTIVATION_FUNCTIONS = [
    'x_tan_x',      # σ(x) = x * tan(x) - specified activation function
    'ReLU',         # Standard ReLU
    'Tanh',         # Hyperbolic tangent
    'Sigmoid',      # Sigmoid
    'Swish',        # Swish activation function
    'GELU'          # GELU activation function
]

# Experiment configuration
EXPERIMENT_CONFIG = {
    'gamma_values': [0.5, 1.0, 1.5, 2.0],  # Different initialization parameters
    'learning_rates': [0.001, 0.01, 0.1],   # Different learning rates
    'hidden_widths': [50, 100, 200]         # Different hidden layer widths
}