"""
Matrix Completion Experiment for Optimistic Estimation

This experiment explores how the specific position distribution of observed data 
affects the sample complexity required for matrix completion tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from scipy.stats import entropy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MATRIX_COMPLETION, RANDOM_SEED
from utils import (
    set_random_seeds, setup_logging, setup_plotting, save_results, save_plot,
    ExperimentTracker, frobenius_norm, relative_error, generate_low_rank_matrix,
    generate_observation_mask, compute_confidence_interval
)


class MatrixCompletionExperiment:
    """
    Matrix Completion Position Analysis Experiment
    
    Explores how the distribution of observed entries affects the sample complexity
    required for successful matrix completion.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the experiment with configuration."""
        self.config = config or MATRIX_COMPLETION
        self.experiment_name = "matrix_completion"
        
        # Setup logging and plotting
        setup_logging(self.experiment_name)
        setup_plotting()
        
        # Set random seeds
        set_random_seeds(RANDOM_SEED)
        
        # Initialize tracker
        self.tracker = ExperimentTracker(self.experiment_name)
        
        # Results storage
        self.results = {
            'position_effects': {},
            'completion_performance': {},
            'pattern_analysis': {},
            'sample_complexity': {}
        }
    
    def generate_target_matrix(self) -> np.ndarray:
        """Generate target low-rank matrix for completion."""
        shape = self.config['matrix_size']
        rank = self.config['rank']
        
        # Generate low-rank matrix
        target_matrix = generate_low_rank_matrix(shape, rank, noise_level=0.01)
        
        return target_matrix
    
    def complete_matrix(self, target_matrix: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Complete matrix using alternating least squares (ALS).
        """
        m, n = target_matrix.shape
        rank = self.config['rank']
        
        # Initialize factors
        U = np.random.randn(m, rank) * self.config['init_scale']
        V = np.random.randn(rank, n) * self.config['init_scale']
        
        # Observed entries
        observed_entries = target_matrix[mask]
        
        learning_rate = self.config['learning_rate']
        max_iterations = self.config['max_iterations']
        tolerance = self.config['tolerance']
        
        losses = []
        errors = []
        
        for iteration in range(max_iterations):
            # Reconstruct matrix
            M_reconstructed = U @ V
            
            # Compute loss on observed entries only
            loss = np.mean((M_reconstructed[mask] - observed_entries) ** 2)
            
            # Compute full matrix error for monitoring
            error = relative_error(target_matrix, M_reconstructed)
            
            losses.append(loss)
            errors.append(error)
            
            # Check convergence
            if loss < tolerance:
                break
            
            # Alternating least squares updates
            # Update U (fix V)
            for i in range(m):
                if np.any(mask[i, :]):
                    observed_cols = mask[i, :]
                    V_obs = V[:, observed_cols]
                    y_obs = target_matrix[i, observed_cols]
                    
                    # Solve least squares: U[i, :] = argmin ||V_obs.T @ U[i, :] - y_obs||^2
                    if V_obs.shape[1] >= rank:
                        U[i, :] = np.linalg.lstsq(V_obs.T, y_obs, rcond=None)[0]
            
            # Update V (fix U)
            for j in range(n):
                if np.any(mask[:, j]):
                    observed_rows = mask[:, j]
                    U_obs = U[observed_rows, :]
                    y_obs = target_matrix[observed_rows, j]
                    
                    # Solve least squares: V[:, j] = argmin ||U_obs @ V[:, j] - y_obs||^2
                    if U_obs.shape[0] >= rank:
                        V[:, j] = np.linalg.lstsq(U_obs, y_obs, rcond=None)[0]
        
        return {
            'U': U,
            'V': V,
            'reconstructed_matrix': U @ V,
            'losses': losses,
            'errors': errors,
            'converged': loss < tolerance,
            'iterations': iteration + 1,
            'final_loss': loss,
            'final_error': error
        }
    
    def analyze_observation_pattern(self, mask: np.ndarray) -> Dict:
        """Analyze properties of the observation pattern."""
        m, n = mask.shape
        
        # Basic statistics
        total_entries = m * n
        observed_entries = np.sum(mask)
        observation_ratio = observed_entries / total_entries
        
        # Row and column coverage
        row_coverage = np.sum(np.any(mask, axis=1)) / m
        col_coverage = np.sum(np.any(mask, axis=0)) / n
        
        # Distribution uniformity (entropy-based measure)
        row_counts = np.sum(mask, axis=1)
        col_counts = np.sum(mask, axis=0)
        
        # Normalize to get probability distributions
        row_dist = row_counts / np.sum(row_counts) if np.sum(row_counts) > 0 else np.zeros_like(row_counts)
        col_dist = col_counts / np.sum(col_counts) if np.sum(col_counts) > 0 else np.zeros_like(col_counts)
        
        # Compute entropy (higher entropy = more uniform distribution)
        row_entropy = entropy(row_dist + 1e-10)  # Add small constant to avoid log(0)
        col_entropy = entropy(col_dist + 1e-10)
        
        # Clustering measure (average distance between observed entries)
        observed_positions = np.where(mask)
        if len(observed_positions[0]) > 1:
            positions = np.column_stack(observed_positions)
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            avg_distance = np.mean(distances)
            clustering_score = 1.0 / (1.0 + avg_distance)  # Higher score = more clustered
        else:
            clustering_score = 1.0
        
        return {
            'observation_ratio': observation_ratio,
            'observed_entries': observed_entries,
            'row_coverage': row_coverage,
            'col_coverage': col_coverage,
            'row_entropy': row_entropy,
            'col_entropy': col_entropy,
            'clustering_score': clustering_score,
            'pattern_uniformity': (row_entropy + col_entropy) / 2
        }
    
    def run_position_effect_analysis(self) -> Dict:
        """Run analysis of how observation positions affect completion performance."""
        logging.info("Running position effect analysis...")
        
        # Generate target matrix
        target_matrix = self.generate_target_matrix()
        
        results = {
            'target_matrix_shape': target_matrix.shape,
            'target_matrix_stats': {
                'mean': float(np.mean(target_matrix)),
                'std': float(np.std(target_matrix)),
                'min': float(np.min(target_matrix)),
                'max': float(np.max(target_matrix))
            },
            'strategies': self.config['position_strategies'],
            'completion_ratios': self.config['completion_ratios'],
            'performance_by_strategy': {},
            'pattern_properties': {},
            'sample_complexity_curves': {}
        }
        
        for strategy in tqdm(self.config['position_strategies'], desc="Testing position strategies"):
            strategy_results = {
                'completion_errors': [],
                'convergence_rates': [],
                'pattern_analyses': [],
                'sample_complexities': []
            }
            
            for ratio in tqdm(self.config['completion_ratios'], desc=f"Strategy: {strategy}", leave=False):
                ratio_errors = []
                ratio_convergence = []
                ratio_patterns = []
                
                for trial in range(self.config['num_trials']):
                    # Generate observation mask
                    mask = generate_observation_mask(target_matrix.shape, ratio, strategy)
                    
                    # Analyze pattern properties
                    pattern_analysis = self.analyze_observation_pattern(mask)
                    ratio_patterns.append(pattern_analysis)
                    
                    # Perform matrix completion
                    completion_result = self.complete_matrix(target_matrix, mask)
                    
                    ratio_errors.append(completion_result['final_error'])
                    ratio_convergence.append(completion_result['converged'])
                
                # Aggregate results for this ratio
                strategy_results['completion_errors'].append(ratio_errors)
                strategy_results['convergence_rates'].append(np.mean(ratio_convergence))
                strategy_results['pattern_analyses'].append(ratio_patterns)
                
                # Determine sample complexity (minimum samples for good performance)
                mean_error = np.mean(ratio_errors)
                if mean_error < 0.1:  # 10% error threshold
                    num_samples = int(ratio * np.prod(target_matrix.shape))
                    strategy_results['sample_complexities'].append(num_samples)
                else:
                    strategy_results['sample_complexities'].append(None)
            
            results['performance_by_strategy'][strategy] = strategy_results
        
        return results
    
    def analyze_pattern_properties(self, position_results: Dict) -> Dict:
        """Analyze the relationship between pattern properties and completion performance."""
        logging.info("Analyzing pattern properties...")
        
        analysis = {
            'property_correlations': {},
            'strategy_rankings': {},
            'optimal_patterns': {}
        }
        
        # Collect all pattern properties and corresponding errors
        all_properties = []
        all_errors = []
        all_strategies = []
        all_ratios = []
        
        for strategy in position_results['strategies']:
            strategy_data = position_results['performance_by_strategy'][strategy]
            
            for ratio_idx, ratio in enumerate(position_results['completion_ratios']):
                errors = strategy_data['completion_errors'][ratio_idx]
                patterns = strategy_data['pattern_analyses'][ratio_idx]
                
                for error, pattern in zip(errors, patterns):
                    all_properties.append(pattern)
                    all_errors.append(error)
                    all_strategies.append(strategy)
                    all_ratios.append(ratio)
        
        # Compute correlations between pattern properties and completion error
        property_names = ['row_coverage', 'col_coverage', 'row_entropy', 'col_entropy', 
                         'clustering_score', 'pattern_uniformity']
        
        for prop_name in property_names:
            prop_values = [p[prop_name] for p in all_properties]
            correlation = np.corrcoef(prop_values, all_errors)[0, 1]
            analysis['property_correlations'][prop_name] = correlation
        
        # Rank strategies by average performance
        strategy_avg_errors = {}
        for strategy in position_results['strategies']:
            strategy_data = position_results['performance_by_strategy'][strategy]
            all_strategy_errors = []
            for errors_list in strategy_data['completion_errors']:
                all_strategy_errors.extend(errors_list)
            strategy_avg_errors[strategy] = np.mean(all_strategy_errors)
        
        # Sort strategies by performance (lower error is better)
        sorted_strategies = sorted(strategy_avg_errors.items(), key=lambda x: x[1])
        analysis['strategy_rankings'] = {rank: (strategy, error) 
                                       for rank, (strategy, error) in enumerate(sorted_strategies)}
        
        # Find optimal patterns for each completion ratio
        for ratio_idx, ratio in enumerate(position_results['completion_ratios']):
            best_strategy = None
            best_error = float('inf')
            
            for strategy in position_results['strategies']:
                strategy_data = position_results['performance_by_strategy'][strategy]
                errors = strategy_data['completion_errors'][ratio_idx]
                mean_error = np.mean(errors)
                
                if mean_error < best_error:
                    best_error = mean_error
                    best_strategy = strategy
            
            analysis['optimal_patterns'][ratio] = {
                'strategy': best_strategy,
                'error': best_error
            }
        
        return analysis
    
    def create_visualizations(self, position_results: Dict, pattern_analysis: Dict):
        """Create comprehensive visualizations."""
        logging.info("Creating visualizations...")
        
        # 1. Performance comparison across strategies and ratios
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Completion error vs observation ratio for different strategies
        ax1 = axes[0, 0]
        for strategy in position_results['strategies']:
            strategy_data = position_results['performance_by_strategy'][strategy]
            mean_errors = [np.mean(errors) for errors in strategy_data['completion_errors']]
            std_errors = [np.std(errors) for errors in strategy_data['completion_errors']]
            
            ax1.errorbar(position_results['completion_ratios'], mean_errors, 
                        yerr=std_errors, marker='o', label=strategy, capsize=5)
        
        ax1.set_xlabel('Observation Ratio')
        ax1.set_ylabel('Completion Error')
        ax1.set_title('Completion Performance vs Observation Strategy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Convergence rates
        ax2 = axes[0, 1]
        for strategy in position_results['strategies']:
            strategy_data = position_results['performance_by_strategy'][strategy]
            convergence_rates = strategy_data['convergence_rates']
            
            ax2.plot(position_results['completion_ratios'], convergence_rates, 
                    marker='s', label=strategy)
        
        ax2.set_xlabel('Observation Ratio')
        ax2.set_ylabel('Convergence Rate')
        ax2.set_title('Convergence Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Property correlations
        ax3 = axes[1, 0]
        properties = list(pattern_analysis['property_correlations'].keys())
        correlations = list(pattern_analysis['property_correlations'].values())
        
        bars = ax3.bar(range(len(properties)), correlations)
        ax3.set_xlabel('Pattern Properties')
        ax3.set_ylabel('Correlation with Error')
        ax3.set_title('Pattern Property vs Performance Correlation')
        ax3.set_xticks(range(len(properties)))
        ax3.set_xticklabels(properties, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Color bars based on correlation strength
        for bar, corr in zip(bars, correlations):
            if abs(corr) > 0.5:
                bar.set_color('red' if corr > 0 else 'green')
            else:
                bar.set_color('gray')
        
        # Plot 4: Strategy ranking
        ax4 = axes[1, 1]
        rankings = pattern_analysis['strategy_rankings']
        strategies = [rankings[i][0] for i in range(len(rankings))]
        avg_errors = [rankings[i][1] for i in range(len(rankings))]
        
        bars = ax4.bar(range(len(strategies)), avg_errors)
        ax4.set_xlabel('Strategy (Ranked by Performance)')
        ax4.set_ylabel('Average Completion Error')
        ax4.set_title('Strategy Performance Ranking')
        ax4.set_xticks(range(len(strategies)))
        ax4.set_xticklabels(strategies, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Color best strategy
        bars[0].set_color('green')
        
        plt.tight_layout()
        save_plot(fig, 'matrix_completion_analysis', self.experiment_name)
        plt.close()
        
        # 2. Pattern visualization examples
        fig, axes = plt.subplots(2, len(position_results['strategies']), 
                                figsize=(4*len(position_results['strategies']), 8))
        
        # Generate a new target matrix for visualization (since we don't store the original)
        target_matrix_vis = self.generate_target_matrix()
        
        for i, strategy in enumerate(position_results['strategies']):
            # Generate example masks for visualization
            mask_low = generate_observation_mask(target_matrix_vis.shape, 0.2, strategy)
            mask_high = generate_observation_mask(target_matrix_vis.shape, 0.6, strategy)
            
            # Plot low ratio mask
            axes[0, i].imshow(mask_low, cmap='Blues', alpha=0.7)
            axes[0, i].set_title(f'{strategy.title()}\n(20% observed)')
            axes[0, i].set_xlabel('Column')
            axes[0, i].set_ylabel('Row')
            
            # Plot high ratio mask
            axes[1, i].imshow(mask_high, cmap='Blues', alpha=0.7)
            axes[1, i].set_title(f'{strategy.title()}\n(60% observed)')
            axes[1, i].set_xlabel('Column')
            axes[1, i].set_ylabel('Row')
        
        plt.tight_layout()
        save_plot(fig, 'observation_patterns', self.experiment_name)
        plt.close()
        
        # 3. Sample complexity comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for strategy in position_results['strategies']:
            strategy_data = position_results['performance_by_strategy'][strategy]
            sample_complexities = strategy_data['sample_complexities']
            
            # Filter out None values
            valid_ratios = []
            valid_complexities = []
            for ratio, complexity in zip(position_results['completion_ratios'], sample_complexities):
                if complexity is not None:
                    valid_ratios.append(ratio)
                    valid_complexities.append(complexity)
            
            if valid_ratios:
                ax.plot(valid_ratios, valid_complexities, marker='o', label=strategy)
        
        ax.set_xlabel('Observation Ratio')
        ax.set_ylabel('Sample Complexity (Minimum Samples)')
        ax.set_title('Sample Complexity vs Observation Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_plot(fig, 'sample_complexity_comparison', self.experiment_name)
        plt.close()
    
    def run_experiment(self) -> Dict:
        """Run the complete matrix completion experiment."""
        self.tracker.start()
        
        logging.info("Starting Matrix Completion Experiment")
        
        # Run position effect analysis
        position_results = self.run_position_effect_analysis()
        self.results['position_effects'] = position_results
        
        # Analyze pattern properties
        pattern_analysis = self.analyze_pattern_properties(position_results)
        self.results['pattern_analysis'] = pattern_analysis
        
        # Create visualizations
        self.create_visualizations(position_results, pattern_analysis)
        
        # Log key findings
        best_strategy = pattern_analysis['strategy_rankings'][0][0]
        best_error = pattern_analysis['strategy_rankings'][0][1]
        
        self.tracker.log_result('best_strategy', best_strategy)
        self.tracker.log_result('best_average_error', best_error)
        
        # Log property correlations
        for prop, corr in pattern_analysis['property_correlations'].items():
            self.tracker.log_result(f'correlation_{prop}', corr)
        
        # Save results
        self.tracker.save_results()
        save_results(self.results, 'matrix_completion_detailed', self.experiment_name)
        
        self.tracker.end()
        
        return self.results


def main():
    """Run the matrix completion experiment."""
    experiment = MatrixCompletionExperiment()
    results = experiment.run_experiment()
    
    print("\n" + "="*50)
    print("MATRIX COMPLETION EXPERIMENT RESULTS")
    print("="*50)
    
    pattern_analysis = results['pattern_analysis']
    
    print("Strategy Performance Ranking:")
    for rank, (strategy, error) in pattern_analysis['strategy_rankings'].items():
        print(f"  {rank + 1}. {strategy}: {error:.4f} average error")
    
    print("\nPattern Property Correlations with Error:")
    for prop, corr in pattern_analysis['property_correlations'].items():
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Weak"
        print(f"  {prop}: {corr:.3f} ({strength} {direction})")
    
    print("\nOptimal Patterns by Observation Ratio:")
    for ratio, info in pattern_analysis['optimal_patterns'].items():
        print(f"  {ratio:.1f}: {info['strategy']} (error: {info['error']:.4f})")
    
    print("="*50)


if __name__ == "__main__":
    main()