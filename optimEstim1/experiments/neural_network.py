"""
Neural Network Complexity Verification Experiment for Optimistic Estimation

This experiment verifies that for more complex target functions, neural networks
require no less than the optimistic sample complexity to recover the target function.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NEURAL_NETWORK, RANDOM_SEED
from utils import (
    set_random_seeds, setup_logging, setup_plotting, save_results, save_plot,
    ExperimentTracker, compute_confidence_interval
)


class NeuralNetworkExperiment:
    """
    Neural Network Complexity Verification Experiment
    
    Verifies that neural networks require at least the optimistic sample complexity
    to recover complex target functions with zero generalization error.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the experiment with configuration."""
        self.config = config or NEURAL_NETWORK
        self.experiment_name = "neural_network"
        
        # Setup logging and plotting
        setup_logging(self.experiment_name)
        setup_plotting()
        
        # Set random seeds
        set_random_seeds(RANDOM_SEED)
        
        # Initialize tracker
        self.tracker = ExperimentTracker(self.experiment_name)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Results storage
        self.results = {
            'target_functions': {},
            'optimistic_bounds': {},
            'neural_network_performance': {},
            'sample_complexity_analysis': {},
            'generalization_analysis': {}
        }
    
    def create_target_functions(self) -> Dict[str, Callable]:
        """Create various target functions of different complexities."""
        target_functions = {}
        
        # Simple polynomial function
        def polynomial_2d(x):
            """Simple 2D polynomial: f(x1, x2) = x1^2 + x2^2 + x1*x2"""
            return x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1]
        
        # Trigonometric function
        def trigonometric_2d(x):
            """2D trigonometric: f(x1, x2) = sin(2*pi*x1) * cos(2*pi*x2)"""
            return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
        
        # Piecewise linear function
        def piecewise_linear_2d(x):
            """2D piecewise linear function"""
            result = np.zeros(x.shape[0])
            mask1 = (x[:, 0] + x[:, 1]) < 0
            mask2 = (x[:, 0] + x[:, 1]) >= 0
            result[mask1] = x[mask1, 0] - x[mask1, 1]
            result[mask2] = -x[mask2, 0] + x[mask2, 1]
            return result
        
        # High-frequency function
        def high_frequency_2d(x):
            """High-frequency 2D function"""
            return np.sin(10 * np.pi * x[:, 0]) * np.sin(10 * np.pi * x[:, 1])
        
        # Composite function
        def composite_2d(x):
            """Composite function combining multiple patterns"""
            poly = x[:, 0]**2 + x[:, 1]**2
            trig = np.sin(4 * np.pi * x[:, 0]) * np.cos(4 * np.pi * x[:, 1])
            return 0.5 * poly + 0.3 * trig + 0.2 * (x[:, 0] * x[:, 1])
        
        target_functions['polynomial'] = polynomial_2d
        target_functions['trigonometric'] = trigonometric_2d
        target_functions['piecewise_linear'] = piecewise_linear_2d
        target_functions['high_frequency'] = high_frequency_2d
        target_functions['composite'] = composite_2d
        
        return target_functions
    
    def compute_optimistic_sample_complexity(self, target_func: Callable, 
                                           input_dim: int = 2) -> Dict:
        """
        Compute theoretical optimistic sample complexity for a target function.
        
        This is a simplified estimation based on function complexity measures.
        """
        # Generate sample points to analyze function properties
        n_analysis_points = 10000
        X_analysis = np.random.uniform(-1, 1, (n_analysis_points, input_dim))
        y_analysis = target_func(X_analysis)
        
        # Compute function properties
        function_variance = np.var(y_analysis)
        function_range = np.max(y_analysis) - np.min(y_analysis)
        
        # Estimate Lipschitz constant (approximate)
        lipschitz_samples = 1000
        X_lip = np.random.uniform(-1, 1, (lipschitz_samples, input_dim))
        y_lip = target_func(X_lip)
        
        lipschitz_estimates = []
        for i in range(min(100, lipschitz_samples)):
            for j in range(i+1, min(i+10, lipschitz_samples)):
                x_diff = np.linalg.norm(X_lip[i] - X_lip[j])
                y_diff = abs(y_lip[i] - y_lip[j])
                if x_diff > 1e-6:
                    lipschitz_estimates.append(y_diff / x_diff)
        
        lipschitz_constant = np.percentile(lipschitz_estimates, 95) if lipschitz_estimates else 1.0
        
        # Estimate effective dimension (based on PCA of function values)
        # This is a heuristic measure
        try:
            from sklearn.decomposition import PCA
            X_grid = np.random.uniform(-1, 1, (1000, input_dim))
            y_grid = target_func(X_grid).reshape(-1, 1)
            
            # Create feature matrix with polynomial features
            X_features = np.column_stack([
                X_grid,
                X_grid**2,
                np.prod(X_grid, axis=1, keepdims=True)
            ])
            
            pca = PCA()
            pca.fit(X_features)
            
            # Effective dimension based on explained variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cumsum_var >= 0.95) + 1
        except:
            effective_dim = input_dim
        
        # Optimistic sample complexity estimation
        # Based on: O(d * log(L/epsilon) / epsilon^2)
        # where d is effective dimension, L is Lipschitz constant, epsilon is target error
        
        target_error = self.config['target_error']
        
        optimistic_complexity = max(
            effective_dim * np.log(lipschitz_constant / target_error) / (target_error**2),
            effective_dim * 10  # Minimum based on dimension
        )
        
        return {
            'optimistic_complexity': int(optimistic_complexity),
            'function_variance': function_variance,
            'function_range': function_range,
            'lipschitz_constant': lipschitz_constant,
            'effective_dimension': effective_dim,
            'target_error': target_error
        }
    
    def create_neural_network(self, input_dim: int, hidden_sizes: List[int]) -> nn.Module:
        """Create a neural network with specified architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        return nn.Sequential(*layers)
    
    def train_neural_network(self, model: nn.Module, X_train: np.ndarray, 
                           y_train: np.ndarray, X_val: np.ndarray = None, 
                           y_val: np.ndarray = None) -> Dict:
        """Train neural network and return training history."""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        model = model.to(self.device)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Training loop
        model.train()
        for epoch in range(self.config['max_epochs']):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_losses.append(val_loss.item())
                model.train()
            
            # Early stopping
            if len(train_losses) > 100 and train_losses[-1] < self.config['tolerance']:
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None,
            'epochs_trained': len(train_losses)
        }
    
    def evaluate_generalization(self, model: nn.Module, target_func: Callable, 
                              input_dim: int = 2, n_test: int = 10000) -> Dict:
        """Evaluate generalization performance on unseen data."""
        
        # Generate test data
        X_test = np.random.uniform(-1, 1, (n_test, input_dim))
        y_test_true = target_func(X_test)
        
        # Predict
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_pred = model(X_test_tensor).cpu().numpy().flatten()
        
        # Compute metrics
        mse = np.mean((y_test_true - y_test_pred)**2)
        mae = np.mean(np.abs(y_test_true - y_test_pred))
        
        # Relative error
        y_range = np.max(y_test_true) - np.min(y_test_true)
        relative_error = np.sqrt(mse) / y_range if y_range > 0 else np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_test_true - y_test_pred)**2)
        ss_tot = np.sum((y_test_true - np.mean(y_test_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'relative_error': relative_error,
            'r_squared': r_squared,
            'rmse': np.sqrt(mse)
        }
    
    def run_sample_complexity_analysis(self, target_func: Callable, 
                                     func_name: str, optimistic_bound: Dict) -> Dict:
        """Run sample complexity analysis for a target function."""
        
        logging.info(f"Running sample complexity analysis for {func_name}")
        
        input_dim = self.config['input_dim']
        sample_sizes = self.config['sample_sizes']
        
        results = {
            'sample_sizes': sample_sizes,
            'generalization_errors': [],
            'training_errors': [],
            'convergence_rates': [],
            'model_complexities': [],
            'optimistic_bound': optimistic_bound['optimistic_complexity']
        }
        
        for sample_size in tqdm(sample_sizes, desc=f"Testing {func_name}"):
            size_errors_gen = []
            size_errors_train = []
            size_convergence = []
            
            for trial in range(self.config['num_trials']):
                # Generate training data
                X_train = np.random.uniform(-1, 1, (sample_size, input_dim))
                y_train = target_func(X_train)
                
                # Split for validation
                if sample_size > 20:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=trial
                    )
                else:
                    X_tr, y_tr = X_train, y_train
                    X_val, y_val = None, None
                
                # Create and train model
                hidden_sizes = self.config['hidden_sizes']
                model = self.create_neural_network(input_dim, hidden_sizes)
                
                training_history = self.train_neural_network(
                    model, X_tr, y_tr, X_val, y_val
                )
                
                # Evaluate generalization
                gen_metrics = self.evaluate_generalization(model, target_func, input_dim)
                
                size_errors_gen.append(gen_metrics['relative_error'])
                size_errors_train.append(training_history['final_train_loss'])
                size_convergence.append(gen_metrics['relative_error'] < self.config['target_error'])
            
            # Aggregate results
            results['generalization_errors'].append(size_errors_gen)
            results['training_errors'].append(size_errors_train)
            results['convergence_rates'].append(np.mean(size_convergence))
            
            # Model complexity (number of parameters)
            model = self.create_neural_network(input_dim, hidden_sizes)
            num_params = sum(p.numel() for p in model.parameters())
            results['model_complexities'].append(num_params)
        
        return results
    
    def analyze_optimistic_bound_verification(self, all_results: Dict) -> Dict:
        """Analyze whether neural networks meet or exceed optimistic bounds."""
        
        analysis = {
            'bound_violations': {},
            'minimum_sample_complexities': {},
            'optimistic_vs_actual': {},
            'complexity_ratios': {}
        }
        
        for func_name, func_results in all_results.items():
            if func_name == 'target_functions':
                continue
                
            sample_sizes = func_results['sample_sizes']
            convergence_rates = func_results['convergence_rates']
            optimistic_bound = func_results['optimistic_bound']
            
            # Find minimum sample size for reliable recovery
            min_sample_size = None
            for i, (size, rate) in enumerate(zip(sample_sizes, convergence_rates)):
                if rate >= 0.8:  # 80% success rate threshold
                    min_sample_size = size
                    break
            
            analysis['minimum_sample_complexities'][func_name] = min_sample_size
            analysis['optimistic_vs_actual'][func_name] = {
                'optimistic': optimistic_bound,
                'actual': min_sample_size,
                'ratio': min_sample_size / optimistic_bound if min_sample_size else float('inf')
            }
            
            # Check if bound is violated (actual < optimistic)
            if min_sample_size and min_sample_size < optimistic_bound:
                analysis['bound_violations'][func_name] = {
                    'violation': True,
                    'violation_ratio': min_sample_size / optimistic_bound
                }
            else:
                analysis['bound_violations'][func_name] = {
                    'violation': False,
                    'violation_ratio': min_sample_size / optimistic_bound if min_sample_size else float('inf')
                }
        
        return analysis
    
    def create_visualizations(self, all_results: Dict, bound_analysis: Dict):
        """Create comprehensive visualizations."""
        
        logging.info("Creating neural network visualizations...")
        
        # 1. Sample complexity curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        func_names = [name for name in all_results.keys() if name != 'target_functions']
        
        for i, func_name in enumerate(func_names):
            if i >= len(axes):
                break
                
            func_results = all_results[func_name]
            sample_sizes = func_results['sample_sizes']
            
            # Plot generalization error
            mean_errors = [np.mean(errors) for errors in func_results['generalization_errors']]
            std_errors = [np.std(errors) for errors in func_results['generalization_errors']]
            
            axes[i].errorbar(sample_sizes, mean_errors, yerr=std_errors, 
                           marker='o', label='Generalization Error', capsize=5)
            
            # Plot optimistic bound
            optimistic_bound = func_results['optimistic_bound']
            axes[i].axvline(x=optimistic_bound, color='red', linestyle='--', 
                          label=f'Optimistic Bound ({optimistic_bound})')
            
            # Plot target error threshold
            axes[i].axhline(y=self.config['target_error'], color='green', 
                          linestyle=':', label='Target Error')
            
            axes[i].set_xlabel('Sample Size')
            axes[i].set_ylabel('Generalization Error')
            axes[i].set_title(f'{func_name.title()} Function')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
            axes[i].set_xscale('log')
        
        plt.tight_layout()
        save_plot(fig, 'neural_network_sample_complexity', self.experiment_name)
        plt.close()
        
        # 2. Convergence rates
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for func_name in func_names:
            func_results = all_results[func_name]
            sample_sizes = func_results['sample_sizes']
            convergence_rates = func_results['convergence_rates']
            
            ax.plot(sample_sizes, convergence_rates, marker='s', label=func_name.title())
        
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Success Rate (Error < Target)')
        ax.set_title('Neural Network Convergence Success Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        save_plot(fig, 'neural_network_convergence_rates', self.experiment_name)
        plt.close()
        
        # 3. Optimistic bound comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        functions = list(bound_analysis['optimistic_vs_actual'].keys())
        optimistic_bounds = [bound_analysis['optimistic_vs_actual'][f]['optimistic'] for f in functions]
        actual_complexities = [bound_analysis['optimistic_vs_actual'][f]['actual'] for f in functions]
        
        # Handle None values
        actual_complexities = [c if c is not None else max(optimistic_bounds) * 2 for c in actual_complexities]
        
        x_pos = np.arange(len(functions))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, optimistic_bounds, width, label='Optimistic Bound', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, actual_complexities, width, label='Actual Complexity', alpha=0.7)
        
        # Color bars based on whether bound is violated
        for i, func in enumerate(functions):
            if bound_analysis['bound_violations'][func]['violation']:
                bars2[i].set_color('red')  # Violation
            else:
                bars2[i].set_color('green')  # No violation
        
        ax.set_xlabel('Target Functions')
        ax.set_ylabel('Sample Complexity')
        ax.set_title('Optimistic Bounds vs Actual Neural Network Complexity')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f.title() for f in functions], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        save_plot(fig, 'optimistic_bound_comparison', self.experiment_name)
        plt.close()
        
        # 4. Violation analysis
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        violation_ratios = [bound_analysis['bound_violations'][f]['violation_ratio'] 
                          for f in functions if bound_analysis['bound_violations'][f]['violation_ratio'] != float('inf')]
        violation_functions = [f for f in functions if bound_analysis['bound_violations'][f]['violation_ratio'] != float('inf')]
        
        if violation_ratios:
            bars = ax.bar(range(len(violation_functions)), violation_ratios)
            
            # Color bars: red if < 1 (violation), green if >= 1 (no violation)
            for bar, ratio in zip(bars, violation_ratios):
                if ratio < 1.0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
            
            ax.axhline(y=1.0, color='black', linestyle='--', label='Optimistic Bound')
            ax.set_xlabel('Target Functions')
            ax.set_ylabel('Actual / Optimistic Ratio')
            ax.set_title('Optimistic Bound Verification\n(Ratio < 1 indicates violation)')
            ax.set_xticks(range(len(violation_functions)))
            ax.set_xticklabels([f.title() for f in violation_functions], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_plot(fig, 'bound_violation_analysis', self.experiment_name)
            plt.close()
    
    def run_experiment(self) -> Dict:
        """Run the complete neural network experiment."""
        
        self.tracker.start()
        logging.info("Starting Neural Network Complexity Verification Experiment")
        
        # Create target functions
        target_functions = self.create_target_functions()
        self.results['target_functions'] = list(target_functions.keys())
        
        # Compute optimistic bounds for each function
        optimistic_bounds = {}
        for func_name, func in target_functions.items():
            bound_info = self.compute_optimistic_sample_complexity(func)
            optimistic_bounds[func_name] = bound_info
            logging.info(f"{func_name}: Optimistic complexity = {bound_info['optimistic_complexity']}")
        
        self.results['optimistic_bounds'] = optimistic_bounds
        
        # Run sample complexity analysis for each function
        all_results = {}
        for func_name, func in target_functions.items():
            func_results = self.run_sample_complexity_analysis(
                func, func_name, optimistic_bounds[func_name]
            )
            all_results[func_name] = func_results
            
            # Log key results
            min_samples = None
            for i, rate in enumerate(func_results['convergence_rates']):
                if rate >= 0.8:
                    min_samples = func_results['sample_sizes'][i]
                    break
            
            self.tracker.log_result(f'{func_name}_optimistic_bound', 
                                  optimistic_bounds[func_name]['optimistic_complexity'])
            self.tracker.log_result(f'{func_name}_min_samples', min_samples or 'Not achieved')
        
        # Analyze optimistic bound verification
        bound_analysis = self.analyze_optimistic_bound_verification(all_results)
        self.results['bound_analysis'] = bound_analysis
        
        # Create visualizations
        self.create_visualizations(all_results, bound_analysis)
        
        # Store detailed results
        self.results['detailed_results'] = all_results
        
        # Save results
        self.tracker.save_results()
        save_results(self.results, 'neural_network_detailed', self.experiment_name)
        
        self.tracker.end()
        
        return self.results


def main():
    """Run the neural network experiment."""
    experiment = NeuralNetworkExperiment()
    results = experiment.run_experiment()
    
    print("\n" + "="*60)
    print("NEURAL NETWORK COMPLEXITY VERIFICATION RESULTS")
    print("="*60)
    
    bound_analysis = results['bound_analysis']
    
    print("Optimistic Bound Verification:")
    print("-" * 40)
    
    for func_name in results['target_functions']:
        optimistic = bound_analysis['optimistic_vs_actual'][func_name]['optimistic']
        actual = bound_analysis['optimistic_vs_actual'][func_name]['actual']
        ratio = bound_analysis['optimistic_vs_actual'][func_name]['ratio']
        violation = bound_analysis['bound_violations'][func_name]['violation']
        
        status = "VIOLATION" if violation else "VERIFIED"
        actual_str = str(actual) if actual is not None else "Not achieved"
        
        print(f"{func_name.title()}:")
        print(f"  Optimistic bound: {optimistic}")
        print(f"  Actual complexity: {actual_str}")
        print(f"  Ratio (Actual/Optimistic): {ratio:.2f}")
        print(f"  Status: {status}")
        print()
    
    # Summary statistics
    violations = sum(1 for f in results['target_functions'] 
                    if bound_analysis['bound_violations'][f]['violation'])
    total_functions = len(results['target_functions'])
    
    print("Summary:")
    print(f"  Total functions tested: {total_functions}")
    print(f"  Bound violations: {violations}")
    print(f"  Bound verification rate: {(total_functions - violations) / total_functions * 100:.1f}%")
    
    if violations == 0:
        print("  ✓ All functions respect the optimistic bound!")
    else:
        print(f"  ⚠ {violations} function(s) violated the optimistic bound")
    
    print("="*60)


if __name__ == "__main__":
    main()