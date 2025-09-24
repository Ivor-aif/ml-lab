"""
Data generator
Used to generate training and test data
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure matplotlib to use ASCII characters only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DataGenerator:
    """Data generator class"""
    
    def __init__(self, boundary=[-1, 1], target_function=None):
        """
        Initialize data generator
        
        Args:
            boundary (list): Data boundary [min, max]
            target_function (callable): Target function
        """
        self.boundary = boundary
        self.target_function = target_function or self._default_target_function
    
    def _default_target_function(self, x):
        """
        Default target function
        f(x) = 0.2 * ReLU(x - 1/3) + 0.2 * ReLU(-x - 1/3)
        """
        return 0.2 * torch.relu(x - 1/3) + 0.2 * torch.relu(-x - 1/3)
    
    def generate_training_data(self, size=100):
        """
        Generate training data
        
        Args:
            size (int): Number of data points
        
        Returns:
            tuple: (input_data, target_data)
        """
        # Uniform sampling within boundary
        x = torch.linspace(self.boundary[0], self.boundary[1], size).unsqueeze(1)
        y = self.target_function(x)
        
        return x, y
    
    def generate_test_data(self, size=1000):
        """
        Generate test data (dense sampling for visualization)
        
        Args:
            size (int): Number of data points
        
        Returns:
            tuple: (input_data, target_data)
        """
        # Dense sampling for smooth curves
        x = torch.linspace(self.boundary[0], self.boundary[1], size).unsqueeze(1)
        y = self.target_function(x)
        
        return x, y
    
    def plot_target_function(self, save_path=None):
        """
        Plot target function
        
        Args:
            save_path (str): Path to save the plot
        """
        x, y = self.generate_test_data(1000)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x.numpy(), y.numpy(), 'b-', linewidth=2, label='Target Function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Target Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
        else:
            plt.show()

class MultiDimDataGenerator:
    """Multi-dimensional data generator"""
    
    def __init__(self, input_dim=5, boundary=[-4, 2]):
        """
        Initialize multi-dimensional data generator
        
        Args:
            input_dim (int): Input dimension
            boundary (list): Data boundary [min, max]
        """
        self.input_dim = input_dim
        self.boundary = boundary
    
    def target_function(self, x):
        """
        Multi-dimensional target function
        f(x) = sum(0.2 * ReLU(x_i - 1/3) + 0.2 * ReLU(-x_i - 1/3))
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1]
        """
        return torch.sum(0.2 * torch.relu(x - 1/3) + 0.2 * torch.relu(-x - 1/3), dim=1, keepdim=True)
    
    def generate_data(self, size=1000):
        """
        Generate multi-dimensional data
        
        Args:
            size (int): Number of data points
        
        Returns:
            tuple: (input_data, target_data)
        """
        # Random sampling within boundary
        x = torch.rand(size, self.input_dim) * (self.boundary[1] - self.boundary[0]) + self.boundary[0]
        y = self.target_function(x)
        
        return x, y

def test_data_generator():
    """Test data generator"""
    print("Testing 1D data generator...")
    
    # Test 1D data generator
    gen = DataGenerator()
    train_x, train_y = gen.generate_training_data(50)
    test_x, test_y = gen.generate_test_data(200)
    
    print(f"Training data shape: {train_x.shape}, {train_y.shape}")
    print(f"Test data shape: {test_x.shape}, {test_y.shape}")
    
    # Plot target function
    gen.plot_target_function()
    
    print("\nTesting multi-dimensional data generator...")
    
    # Test multi-dimensional data generator
    multi_gen = MultiDimDataGenerator(input_dim=3)
    multi_x, multi_y = multi_gen.generate_data(100)
    
    print(f"Multi-dim data shape: {multi_x.shape}, {multi_y.shape}")
    print(f"Sample input: {multi_x[0]}")
    print(f"Sample output: {multi_y[0]}")
    
    print("Data generator test completed!")

if __name__ == "__main__":
    test_data_generator()