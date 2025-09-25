"""
Matrix Factorization Experiment for Optimistic Estimation

This experiment verifies that small initialization enables achieving optimistic 
sample complexity in 5×5 matrix factorization tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MATRIX_FACTORIZATION, RANDOM_SEED
from utils import (
    set_random_seeds, setup_logging, setup_plotting, save_results, save_plot,
    ExperimentTracker, frobenius_norm, relative_error, compute_confidence_interval
)


class MatrixFactorizationExperiment:
    """
    5×5 Matrix Factorization Experiment
    
    Verifies that small initialization enables achieving optimistic sample complexity
    by factorizing a 5×5 matrix into two smaller matrices and analyzing the 
    relationship between initialization scale and sample complexity.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the experiment with configuration."""
        self.config = config or MATRIX_FACTORIZATION
        self.experiment_name = "matrix_factorization"
        
        # Setup logging and plotting
        setup_logging(self.experiment_name)
        setup_plotting()
        
        # Set random seeds
        set_random_seeds(RANDOM_SEED)
        
        # Initialize tracker
        self.tracker = ExperimentTracker(self.experiment_name)
        
        # Results storage
        self.results = {
            'sample_complexity': {},
            'convergence_analysis': {},
            'initialization_effects': {},
            'optimistic_bounds': {}
        }
    
    def generate_target_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate target matrix and its true factorization."""
        size = self.config['matrix_size']
        rank = self.config['rank']
        
        # Generate true factors
        U_true = np.random.randn(size, rank)
        V_true = np.random.randn(rank, size)
        
        # Target matrix
        M_true = U_true @ V_true
        
        # Add small amount of noise
        noise = np.random.randn(size, size) * self.config['noise_level']
        M_target = M_true + noise
        
        return M_target, U_true, V_true
    
    def sample_observations(self, M_target: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample observations from the target matrix."""
        size = self.config['matrix_size']
        total_entries = size * size
        
        # Ensure we don't sample more than available entries
        num_samples = min(num_samples, total_entries)
        
        # Random sampling of matrix entries
        indices = np.random.choice(total_entries, num_samples, replace=False)
        rows, cols = np.unravel_index(indices, (size, size))
        
        observations = M_target[rows, cols]
        
        return (rows, cols), observations
    
    def initialize_factors(self, init_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize factor matrices with given scale."""
        size = self.config['matrix_size']
        rank = self.config['rank']
        
        U = np.random.randn(size, rank) * init_scale
        V = np.random.randn(rank, size) * init_scale
        
        return U, V
    
    def compute_optimistic_sample_complexity(self, M_target: np.ndarray) -> int:
        """
        Compute theoretical optimistic sample complexity.
        
        For matrix factorization, the optimistic sample complexity is approximately
        the number of degrees of freedom in the factorization.
        """
        size = self.config['matrix_size']
        rank = self.config['rank']
        
        # Degrees of freedom: size*rank + rank*size - rank^2 (to account for rotation invariance)
        dof = size * rank + rank * size - rank * rank
        
        # Optimistic bound considers the condition number and noise level
        condition_number = np.linalg.cond(M_target)
        noise_factor = 1 + self.config['noise_level'] * condition_number
        
        optimistic_complexity = int(dof * noise_factor)
        
        return max(optimistic_complexity, dof)  # At least the degrees of freedom
    
    def factorize_matrix(self, M_target: np.ndarray, indices: Tuple[np.ndarray, np.ndarray], 
                        observations: np.ndarray, init_scale: float) -> Dict:
        """
        Perform matrix factorization using gradient descent.
        """
        rows, cols = indices
        U, V = self.initialize_factors(init_scale)
        
        learning_rate = self.config['learning_rate']
        max_iterations = self.config['max_iterations']
        tolerance = self.config['tolerance']
        
        losses = []
        errors = []
        
        for iteration in range(max_iterations):
            # Forward pass
            predictions = np.sum(U[rows, :] * V[:, cols].T, axis=1)
            loss = np.mean((predictions - observations) ** 2)
            
            # Compute full matrix error for monitoring
            M_reconstructed = U @ V
            error = relative_error(M_target, M_reconstructed)
            
            losses.append(loss)
            errors.append(error)
            
            # Check convergence
            if loss < tolerance:
                break
            
            # Backward pass
            residuals = predictions - observations
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            
            for i, (r, c) in enumerate(zip(rows, cols)):
                dU[r, :] += 2 * residuals[i] * V[:, c] / len(observations)
                dV[:, c] += 2 * residuals[i] * U[r, :] / len(observations)
            
            # Update parameters
            U -= learning_rate * dU
            V -= learning_rate * dV
        
        return {
            'U_shape': U.shape,
            'V_shape': V.shape,
            'losses': losses,
            'errors': errors,
            'converged': loss < tolerance,
            'iterations': iteration + 1,
            'final_loss': loss,
            'final_error': error
        }
    
    def run_sample_complexity_analysis(self) -> Dict:
        """Run sample complexity analysis for different initialization scales."""
        logging.info("Running sample complexity analysis...")
        
        # Generate target matrix
        M_target, U_true, V_true = self.generate_target_matrix()
        optimistic_bound = self.compute_optimistic_sample_complexity(M_target)
        
        results = {
            'target_matrix_shape': M_target.shape,
            'target_matrix_stats': {
                'mean': float(np.mean(M_target)),
                'std': float(np.std(M_target)),
                'min': float(np.min(M_target)),
                'max': float(np.max(M_target))
            },
            'true_factors_shapes': (U_true.shape, V_true.shape),
            'optimistic_bound': optimistic_bound,
            'init_scales': self.config['init_scales'],
            'sample_sizes': self.config['sample_sizes'],
            'complexity_curves': {},
            'success_rates': {},
            'convergence_stats': {}
        }
        
        for init_scale in tqdm(self.config['init_scales'], desc="Testing initialization scales"):
            complexity_curve = []
            success_rates = []
            convergence_times = []
            
            for num_samples in tqdm(self.config['sample_sizes'], desc=f"Scale {init_scale}", leave=False):
                trial_errors = []
                trial_successes = []
                trial_convergence = []
                
                for trial in range(self.config['num_trials']):
                    # Sample observations
                    indices, observations = self.sample_observations(M_target, num_samples)
                    
                    # Perform factorization
                    result = self.factorize_matrix(M_target, indices, observations, init_scale)
                    
                    trial_errors.append(result['final_error'])
                    trial_successes.append(result['converged'])
                    trial_convergence.append(result['iterations'])
                
                # Aggregate trial results
                mean_error = np.mean(trial_errors)
                success_rate = np.mean(trial_successes)
                mean_convergence = np.mean(trial_convergence)
                
                complexity_curve.append(mean_error)
                success_rates.append(success_rate)
                convergence_times.append(mean_convergence)
            
            results['complexity_curves'][init_scale] = complexity_curve
            results['success_rates'][init_scale] = success_rates
            results['convergence_stats'][init_scale] = convergence_times
        
        return results
    
    def analyze_initialization_effects(self, sample_complexity_results: Dict) -> Dict:
        """Analyze the effects of different initialization scales."""
        logging.info("Analyzing initialization effects...")
        
        init_scales = sample_complexity_results['init_scales']
        sample_sizes = sample_complexity_results['sample_sizes']
        optimistic_bound = sample_complexity_results['optimistic_bound']
        
        analysis = {
            'optimal_init_scale': None,
            'sample_efficiency': {},
            'initialization_comparison': {},
            'optimistic_achievement': {}
        }
        
        # Find sample size needed to achieve good performance for each init scale
        threshold_error = 0.1  # 10% relative error threshold
        
        for init_scale in init_scales:
            complexity_curve = sample_complexity_results['complexity_curves'][init_scale]
            
            # Find minimum sample size to achieve threshold
            sample_efficiency = None
            for i, error in enumerate(complexity_curve):
                if error < threshold_error:
                    sample_efficiency = sample_sizes[i]
                    break
            
            analysis['sample_efficiency'][init_scale] = sample_efficiency
            
            # Check if optimistic bound is achieved
            if sample_efficiency and sample_efficiency <= optimistic_bound:
                analysis['optimistic_achievement'][init_scale] = True
            else:
                analysis['optimistic_achievement'][init_scale] = False
        
        # Find optimal initialization scale
        valid_scales = {k: v for k, v in analysis['sample_efficiency'].items() if v is not None}
        if valid_scales:
            analysis['optimal_init_scale'] = min(valid_scales, key=valid_scales.get)
        
        return analysis
    
    def create_visualizations(self, sample_complexity_results: Dict, initialization_analysis: Dict):
        """Create comprehensive visualizations."""
        logging.info("Creating visualizations...")
        
        # 1. Sample Complexity Curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sample complexity curves for different initialization scales
        ax1 = axes[0, 0]
        for init_scale in sample_complexity_results['init_scales']:
            complexity_curve = sample_complexity_results['complexity_curves'][init_scale]
            ax1.plot(sample_complexity_results['sample_sizes'], complexity_curve, 
                    marker='o', label=f'Init scale: {init_scale}')
        
        ax1.axvline(x=sample_complexity_results['optimistic_bound'], 
                   color='red', linestyle='--', label='Optimistic bound')
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Relative Error')
        ax1.set_title('Sample Complexity vs Initialization Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Success rates
        ax2 = axes[0, 1]
        for init_scale in sample_complexity_results['init_scales']:
            success_rates = sample_complexity_results['success_rates'][init_scale]
            ax2.plot(sample_complexity_results['sample_sizes'], success_rates, 
                    marker='s', label=f'Init scale: {init_scale}')
        
        ax2.axvline(x=sample_complexity_results['optimistic_bound'], 
                   color='red', linestyle='--', label='Optimistic bound')
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Convergence Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample efficiency comparison
        ax3 = axes[1, 0]
        init_scales = list(initialization_analysis['sample_efficiency'].keys())
        efficiencies = [initialization_analysis['sample_efficiency'][scale] for scale in init_scales]
        
        # Handle None values
        valid_indices = [i for i, eff in enumerate(efficiencies) if eff is not None]
        valid_scales = [init_scales[i] for i in valid_indices]
        valid_efficiencies = [efficiencies[i] for i in valid_indices]
        
        if valid_scales:
            bars = ax3.bar(range(len(valid_scales)), valid_efficiencies)
            ax3.axhline(y=sample_complexity_results['optimistic_bound'], 
                       color='red', linestyle='--', label='Optimistic bound')
            ax3.set_xlabel('Initialization Scale')
            ax3.set_ylabel('Sample Efficiency')
            ax3.set_title('Sample Efficiency vs Initialization Scale')
            ax3.set_xticks(range(len(valid_scales)))
            ax3.set_xticklabels([f'{scale:.0e}' for scale in valid_scales])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Color bars based on optimistic achievement
            for i, (scale, bar) in enumerate(zip(valid_scales, bars)):
                if initialization_analysis['optimistic_achievement'][scale]:
                    bar.set_color('green')
                else:
                    bar.set_color('orange')
        
        # Plot 4: Convergence time analysis
        ax4 = axes[1, 1]
        for init_scale in sample_complexity_results['init_scales']:
            convergence_times = sample_complexity_results['convergence_stats'][init_scale]
            ax4.plot(sample_complexity_results['sample_sizes'], convergence_times, 
                    marker='^', label=f'Init scale: {init_scale}')
        
        ax4.set_xlabel('Number of Samples')
        ax4.set_ylabel('Convergence Iterations')
        ax4.set_title('Convergence Speed Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'matrix_factorization_analysis', self.experiment_name)
        plt.close()
        
        # 2. Target matrix and reconstruction visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Generate a new target matrix for visualization (since we don't store the original)
        M_target_vis, _, _ = self.generate_target_matrix()
        
        # Original matrix
        im1 = axes[0].imshow(M_target_vis, cmap='viridis')
        axes[0].set_title('Target Matrix')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[0])
        
        # Best reconstruction (smallest init scale that achieves optimistic bound)
        best_scale = initialization_analysis['optimal_init_scale']
        if best_scale:
            # Reconstruct with best parameters
            indices, observations = self.sample_observations(M_target_vis, sample_complexity_results['optimistic_bound'])
            result = self.factorize_matrix(M_target_vis, indices, observations, best_scale)
            
            # Reconstruct the matrix using the original factors (need to re-run factorization to get U, V)
            U, V = self.initialize_factors(best_scale)
            rows, cols = indices
            
            # Quick factorization to get the factors
            learning_rate = self.config['learning_rate']
            max_iterations = min(100, self.config['max_iterations'])  # Limit iterations for visualization
            tolerance = self.config['tolerance']
            
            for iteration in range(max_iterations):
                predictions = np.sum(U[rows, :] * V[:, cols].T, axis=1)
                loss = np.mean((predictions - observations) ** 2)
                
                if loss < tolerance:
                    break
                
                residuals = predictions - observations
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                
                for i, (r, c) in enumerate(zip(rows, cols)):
                    dU[r, :] += 2 * residuals[i] * V[:, c] / len(observations)
                    dV[:, c] += 2 * residuals[i] * U[r, :] / len(observations)
                
                U -= learning_rate * dU
                V -= learning_rate * dV
            
            M_reconstructed = U @ V
            
            im2 = axes[1].imshow(M_reconstructed, cmap='viridis')
            axes[1].set_title(f'Best Reconstruction (scale={best_scale:.0e})')
            axes[1].set_xlabel('Column')
            axes[1].set_ylabel('Row')
            plt.colorbar(im2, ax=axes[1])
            
            # Error matrix
            error_matrix = np.abs(M_target_vis - M_reconstructed)
            im3 = axes[2].imshow(error_matrix, cmap='Reds')
            axes[2].set_title('Absolute Error')
            axes[2].set_xlabel('Column')
            axes[2].set_ylabel('Row')
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        save_plot(fig, 'matrix_reconstruction_comparison', self.experiment_name)
        plt.close()
    
    def run_experiment(self) -> Dict:
        """Run the complete matrix factorization experiment."""
        self.tracker.start()
        
        logging.info("Starting Matrix Factorization Experiment")
        
        # Run sample complexity analysis
        sample_complexity_results = self.run_sample_complexity_analysis()
        self.results['sample_complexity'] = sample_complexity_results
        
        # Analyze initialization effects
        initialization_analysis = self.analyze_initialization_effects(sample_complexity_results)
        self.results['initialization_effects'] = initialization_analysis
        
        # Create visualizations
        self.create_visualizations(sample_complexity_results, initialization_analysis)
        
        # Log key findings
        optimistic_bound = sample_complexity_results['optimistic_bound']
        optimal_scale = initialization_analysis['optimal_init_scale']
        
        self.tracker.log_result('optimistic_bound', optimistic_bound)
        self.tracker.log_result('optimal_init_scale', optimal_scale)
        
        if optimal_scale:
            optimal_efficiency = initialization_analysis['sample_efficiency'][optimal_scale]
            self.tracker.log_result('optimal_sample_efficiency', optimal_efficiency)
            
            achieves_optimistic = initialization_analysis['optimistic_achievement'][optimal_scale]
            self.tracker.log_result('achieves_optimistic_bound', achieves_optimistic)
        
        # Save results
        self.tracker.save_results()
        save_results(self.results, 'matrix_factorization_detailed', self.experiment_name)
        
        self.tracker.end()
        
        return self.results


def main():
    """Run the matrix factorization experiment."""
    experiment = MatrixFactorizationExperiment()
    results = experiment.run_experiment()
    
    print("\n" + "="*50)
    print("MATRIX FACTORIZATION EXPERIMENT RESULTS")
    print("="*50)
    
    print(f"Optimistic Sample Complexity Bound: {results['sample_complexity']['optimistic_bound']}")
    
    optimal_scale = results['initialization_effects']['optimal_init_scale']
    if optimal_scale:
        print(f"Optimal Initialization Scale: {optimal_scale:.0e}")
        
        efficiency = results['initialization_effects']['sample_efficiency'][optimal_scale]
        print(f"Sample Efficiency: {efficiency}")
        
        achieves_bound = results['initialization_effects']['optimistic_achievement'][optimal_scale]
        print(f"Achieves Optimistic Bound: {achieves_bound}")
        
        if achieves_bound:
            print("✓ SUCCESS: Small initialization enables achieving optimistic sample complexity!")
        else:
            print("✗ The experiment did not achieve the optimistic bound.")
    else:
        print("✗ No initialization scale achieved the convergence threshold.")
    
    print("="*50)


if __name__ == "__main__":
    main()