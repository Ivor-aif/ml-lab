"""
Trainer module
Responsible for neural network training process and experiment recording
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# Configure matplotlib to use ASCII characters only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class CondenseTrainer:
    """Condensation phenomenon experiment trainer"""
    
    def __init__(self, model, train_data, test_data, config, save_dir):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            train_data: Training data (x, y)
            test_data: Test data (x, y)
            config: Configuration parameters
            save_dir: Save directory
        """
        self.model = model
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data
        self.config = config
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device(config.device if hasattr(config, 'device') 
                                 else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.train_x, self.train_y = self.train_x.to(self.device), self.train_y.to(self.device)
        self.test_x, self.test_y = self.test_x.to(self.device), self.test_y.to(self.device)
        
        # Set optimizer and loss function
        if config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)
        elif config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        self.criterion = nn.MSELoss()
        
        # Record training history
        self.train_losses = []
        self.test_losses = []
        self.similarity_history = []
        self.checkpoints = []
        
        # Save initial model state
        self.initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def train_one_epoch(self):
        """Train one epoch with gradient clipping for numerical stability"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(self.train_x)
        loss = self.criterion(outputs, self.train_y)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping this epoch")
            return float('inf')
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for numerical stability (especially for x_tan_x)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self):
        """Evaluate model"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_x)
            loss = self.criterion(outputs, self.test_y)
        
        return loss.item(), outputs
    
    def train(self, similarity_analyzer=None):
        """
        Complete training process
        
        Args:
            similarity_analyzer: Similarity analyzer
        
        Returns:
            dict: Training results
        """
        print(f"Starting training - Activation: {self.model.activation_name}, Gamma: {self.model.gamma}")
        print(f"Epochs: {self.config.epochs}, Learning rate: {self.config.lr}")
        
        # Record initial similarity
        if similarity_analyzer:
            initial_similarities = similarity_analyzer.compute_parameter_similarity(self.model)
            self.similarity_history.append(initial_similarities['extended_similarity'])
        
        # Training loop
        for epoch in tqdm(range(self.config.epochs), desc="Training progress"):
            # Train one epoch
            train_loss = self.train_one_epoch()
            
            # Evaluate
            test_loss, outputs = self.evaluate()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            # Periodically record similarity and save checkpoints
            if (epoch + 1) % self.config.save_epoch == 0:
                # Record similarity
                if similarity_analyzer:
                    similarities = similarity_analyzer.compute_parameter_similarity(self.model)
                    self.similarity_history.append(similarities['extended_similarity'])
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': {k: v.clone() for k, v in self.model.state_dict().items()},
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                self.checkpoints.append(checkpoint)
                
                # Print progress
                if (epoch + 1) % self.config.plot_epoch == 0:
                    print(f"Epoch [{epoch+1}/{self.config.epochs}] - "
                          f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Record final similarity
        if similarity_analyzer:
            final_similarities = similarity_analyzer.compute_parameter_similarity(self.model)
            self.similarity_history.append(final_similarities['extended_similarity'])
        
        print("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'similarity_history': self.similarity_history,
            'final_output': outputs,
            'model_state': self.model.state_dict(),
            'initial_state': self.initial_state
        }
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(self.test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Log-scale loss curves
        plt.subplot(1, 2, 2)
        plt.loglog(self.train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.loglog(self.test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
        else:
            plt.show()
    
    def plot_function_fitting(self, save_path=None):
        """Plot function fitting results"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_x)
        
        plt.figure(figsize=(10, 6))
        
        # Convert to numpy for plotting
        test_x_np = self.test_x.cpu().numpy()
        test_y_np = self.test_y.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        train_x_np = self.train_x.cpu().numpy()
        train_y_np = self.train_y.cpu().numpy()
        
        # Sort for continuous curve plotting
        if test_x_np.shape[1] == 1:  # 1D input
            sort_idx = np.argsort(test_x_np.flatten())
            test_x_sorted = test_x_np[sort_idx]
            test_y_sorted = test_y_np[sort_idx]
            outputs_sorted = outputs_np[sort_idx]
            
            plt.plot(test_x_sorted, test_y_sorted, 'r-', linewidth=2, label='True Function')
            plt.plot(test_x_sorted, outputs_sorted, 'b--', linewidth=2, label='Network Output')
            plt.scatter(train_x_np, train_y_np, c='green', s=50, 
                       marker='o', label='Training Data', zorder=5)
            
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'Function Fitting Results - {self.model.activation_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Multi-dimensional input, can only show scatter plot
            plt.scatter(range(len(test_y_np)), test_y_np.flatten(), 
                       c='red', alpha=0.6, label='True Values')
            plt.scatter(range(len(outputs_np)), outputs_np.flatten(), 
                       c='blue', alpha=0.6, label='Predicted Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Function Value')
            plt.title(f'Function Fitting Results - {self.model.activation_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
        else:
            plt.show()
    
    def save_results(self, results, filename=None):
        """Save experiment results"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        save_path = os.path.join(self.save_dir, filename)
        
        # Prepare data for saving (convert tensors to lists)
        save_data = {
            'config': {
                'activation': self.model.activation_name,
                'gamma': self.model.gamma,
                'lr': self.config.lr,
                'epochs': self.config.epochs,
                'hidden_width': self.model.hidden_layers_width,
                'optimizer': self.config.optimizer
            },
            'train_losses': results['train_losses'],
            'test_losses': results['test_losses'],
            'final_train_loss': results['train_losses'][-1],
            'final_test_loss': results['test_losses'][-1]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment results saved to: {save_path}")
        
        return save_path

def test_trainer():
    """Test trainer"""
    from model import CondenseNet, create_target_function
    from data_generator import DataGenerator
    from config import get_config
    
    # Get configuration
    config = get_config()
    config.epochs = 100  # Reduce epochs for testing
    
    # Generate data
    data_gen = DataGenerator()
    train_x, train_y = data_gen.generate_training_data(50)
    test_x, test_y = data_gen.generate_test_data(200)
    
    # Create model
    model = CondenseNet(input_dim=1, output_dim=1, hidden_layers_width=[20], 
                       activation='ReLU', gamma=1.0)
    
    # Create trainer
    trainer = CondenseTrainer(model, (train_x, train_y), (test_x, test_y), config)
    
    # Train
    results = trainer.train()
    
    # Plot results
    trainer.plot_training_curves()
    trainer.plot_function_fitting()
    
    # Save results
    trainer.save_results(results)
    
    print("Trainer test completed!")

if __name__ == "__main__":
    test_trainer()