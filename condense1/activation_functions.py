"""
Custom activation functions module
Contains implementations of various activation functions, especially σ(x) = x * tan(x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure matplotlib to use ASCII characters only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class XTanX(nn.Module):
    """
    Custom activation function: σ(x) = x * tan(x)
    Enhanced with numerical stability improvements
    """
    def __init__(self, clamp_range=1.4):
        super(XTanX, self).__init__()
        self.clamp_range = clamp_range  # Slightly less than π/2 ≈ 1.57
    
    def forward(self, x):
        # Enhanced numerical stability
        # 1. Clamp input to avoid tan(x) singularities
        x_clamped = torch.clamp(x, -self.clamp_range, self.clamp_range)
        
        # 2. Compute tan(x) with numerical stability checks
        tan_x = torch.tan(x_clamped)
        
        # 3. Check for NaN or infinite values and replace with safe values
        result = x_clamped * tan_x
        
        # 4. Replace any NaN or infinite values with linear approximation
        mask_invalid = ~torch.isfinite(result)
        if mask_invalid.any():
            # For small x, x*tan(x) ≈ x (since tan(x) ≈ x for small x)
            result = torch.where(mask_invalid, x_clamped, result)
        
        return result

class Swish(nn.Module):
    """
    Swish activation function: σ(x) = x * sigmoid(x)
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class GELU(nn.Module):
    """
    GELU activation function
    """
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, x):
        return F.gelu(x)

class XTanhX(nn.Module):
    """
    Custom activation function: σ(x) = x * tanh(x)
    """
    def __init__(self):
        super(XTanhX, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(x)

def get_activation_function(name):
    """
    Get activation function by name
    
    Args:
        name (str): Activation function name
    
    Returns:
        nn.Module: Activation function module
    """
    activation_dict = {
        'x_tan_x': XTanX(),
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Swish': Swish(),
        'GELU': GELU(),
        'x_tanh_x': XTanhX(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'ELU': nn.ELU()
    }
    
    if name not in activation_dict:
        raise ValueError(f"Unknown activation function: {name}. Available functions: {list(activation_dict.keys())}")
    
    return activation_dict[name]

def plot_activation_functions():
    """
    Plot various activation functions
    """
    x = torch.linspace(-2, 2, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    activation_names = ['x_tan_x', 'ReLU', 'Tanh', 'Sigmoid', 'Swish', 'GELU']
    
    for i, name in enumerate(activation_names):
        try:
            act_func = get_activation_function(name)
            with torch.no_grad():
                y = act_func(x)
            
            axes[i].plot(x.numpy(), y.numpy(), linewidth=2)
            axes[i].set_title(f'{name}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{name} (Error)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print("Activation functions plot saved as 'activation_functions.png'")

if __name__ == "__main__":
    print("Testing activation functions...")
    plot_activation_functions()