"""
Neural network model definition
Contains fully connected network and parameter initialization methods
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List
from activation_functions import get_activation_function

class CondenseNet(nn.Module):
    """
    Fully connected neural network for condensation phenomenon experiments
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_layers_width=[100], 
                 activation='ReLU', gamma=1.0):
        """
        Initialize network
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            hidden_layers_width (list): Hidden layer width list
            activation (str): Activation function name
            gamma (float): Initialization parameter
        """
        super(CondenseNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers_width = hidden_layers_width
        self.activation_name = activation
        self.gamma = gamma
        
        # Build network layers
        layers = []
        layer_widths = [input_dim] + hidden_layers_width
        
        # Hidden layers
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            layers.append(get_activation_function(activation))
        
        # Output layer (no activation function)
        layers.append(nn.Linear(hidden_layers_width[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize parameters
        self._initialize_weights()
    
    def forward(self, x):
        """Forward propagation"""
        return self.network(x)
    
    def _initialize_weights(self):
        """
        Initialize network weights with specified gamma parameter
        Parameter initialization: θ ~ N(0, 1/m^γ)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Get layer width (input dimension)
                fan_in = module.weight.size(1)
                
                # Calculate initialization standard deviation
                std = 1.0 / (fan_in ** self.gamma)
                
                # Initialize weights and biases
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=std)
    
    def get_hidden_weights(self):
        """
        Get weights and biases of the first hidden layer
        Used to calculate neuron directions and amplitudes
        
        Returns:
            tuple: (weights, biases, output_weights)
        """
        # Get first linear layer (input to hidden)
        first_layer = self.network[0]
        weights = first_layer.weight.data  # shape: [hidden_size, input_dim]
        biases = first_layer.bias.data     # shape: [hidden_size]
        
        # Get last linear layer (hidden to output)
        last_layer = None
        for module in self.network:
            if isinstance(module, nn.Linear):
                last_layer = module
        
        output_weights = last_layer.weight.data  # shape: [output_dim, hidden_size]
        
        return weights, biases, output_weights
    
    def get_neuron_features(self):
        """
        Calculate neuron features (directions and amplitudes)
        
        Returns:
            tuple: (orientations, amplitudes)
        """
        weights, biases, output_weights = self.get_hidden_weights()
        
        # For 1D input, calculate directions and amplitudes
        if self.input_dim == 1:
            # Extend weight vector: w_k = [w_k, b_k]
            extended_weights = torch.stack([weights.squeeze(), biases], dim=1)  # [hidden_size, 2]
            
            # Calculate L2 norm
            norms = torch.norm(extended_weights, dim=1)  # [hidden_size]
            
            # Calculate directions (angles)
            w_normalized = weights.squeeze() / norms
            b_normalized = biases / norms
            orientations = torch.sign(b_normalized) * torch.acos(torch.clamp(w_normalized, -1, 1))
            
            # Calculate amplitudes
            amplitudes = torch.abs(output_weights.squeeze()) * norms
            
            return orientations, amplitudes
        else:
            # For high-dimensional input, return weight vector norms as amplitudes
            norms = torch.norm(weights, dim=1)
            amplitudes = torch.abs(output_weights.squeeze()) * norms
            return weights, amplitudes

def create_target_function(x):
    """
    Create target function
    f(x) = 0.2 * ReLU(x - 1/3) + 0.2 * ReLU(-x - 1/3)
    
    Args:
        x (torch.Tensor): Input data
    
    Returns:
        torch.Tensor: Target function values
    """
    return 0.2 * torch.relu(x - 1/3) + 0.2 * torch.relu(-x - 1/3)

def test_model():
    """Test model"""
    # Create test data
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = create_target_function(x)
    
    # Create model
    model = CondenseNet(input_dim=1, output_dim=1, hidden_layers_width=[50], 
                       activation='x_tan_x', gamma=1.0)
    
    # Forward propagation
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {y.shape}")
    
    # Test feature extraction
    orientations, amplitudes = model.get_neuron_features()
    print(f"Orientations shape: {orientations.shape}")
    print(f"Amplitudes shape: {amplitudes.shape}")

if __name__ == "__main__":
    test_model()