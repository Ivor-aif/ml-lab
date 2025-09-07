"""Complete Frequency Analysis Experiment

This module implements a comprehensive two-step frequency analysis experiment:
1. Step 1: Neural network learns to fit frequency functions from sample points
2. Step 2: Neural network learns to predict frequency parameters directly from data points

Features:
- Enhanced visualization with training evolution comparison
- Function fitting quality comparison
- Parameter prediction accuracy analysis
- Original vs predicted function comparison
- Comprehensive performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from config import ExperimentConfig as Config
from data_generator import FrequencyDataGenerator


class CompleteFrequencyExperiment:
    """Complete frequency analysis experiment class"""
    
    def __init__(self, config: Config):
        """Initialize experiment
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.generator = FrequencyDataGenerator(config)
        
        # Results storage
        self.step1_results = {}
        self.step2_results = {}
        self.training_history = {'step1': [], 'step2': []}
        
        # Models
        self.step1_model = None
        self.step2_model = None
        
        # Scalers
        self.step1_scaler_x = StandardScaler()
        self.step1_scaler_y = StandardScaler()
        self.step2_scaler_x = StandardScaler()
        self.step2_scaler_y = StandardScaler()
        
        # Data storage
        self.experiment_data = []
        
    def generate_random_parameters(self, num_components: int = 3) -> Dict:
        """Generate random frequency parameters
        
        Args:
            num_components: Number of frequency components
            
        Returns:
            Dictionary of parameters
        """
        params = {
            'a0': np.random.uniform(-2, 2)
        }
        
        for i in range(1, num_components + 1):
            params[f'a{i}'] = np.random.uniform(-1, 1)
            params[f'b{i}'] = np.random.uniform(0, 2*np.pi)
        
        return params
    
    def generate_experiment_data(self, num_experiments: int = 100) -> List[Dict]:
        """Generate multiple experiment datasets
        
        Args:
            num_experiments: Number of experiments to generate
            
        Returns:
            List of experiment data dictionaries
        """
        print(f"Generating {num_experiments} experiment samples...")
        
        experiments = []
        for i in range(num_experiments):
            # Generate random parameters
            true_params = self.generate_random_parameters()
            
            # Update generator coefficients
            self.generator.coefficients = true_params
            self.generator.frequency_components = [
                (true_params[f'a{j}'], true_params[f'b{j}'], j) 
                for j in range(1, 4)  # 3 frequency components
            ]
            
            # Generate sample points
            x_samples, y_samples = self.generator.generate_samples(
                num_samples=self.config.data_params['num_samples'],
                x_range=self.config.data_params['x_range'],
                noise_level=self.config.data_params['noise_level']
            )
            
            experiments.append({
                'id': i,
                'true_params': true_params,
                'x_samples': x_samples,
                'y_samples': y_samples
            })
            
        self.experiment_data = experiments
        return experiments
    
    def run_step1_experiment(self) -> Dict[str, Any]:
        """Run Step 1: Function fitting experiment
        
        Returns:
            Step 1 results dictionary
        """
        print("\n=== Step 1: Function Fitting Experiment ===")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for exp in self.experiment_data:
            X_train.append(exp['x_samples'])
            y_train.append(exp['y_samples'])
        
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train)
        
        # Scale data
        X_train_scaled = self.step1_scaler_x.fit_transform(X_train)
        y_train_scaled = self.step1_scaler_y.fit_transform(y_train)
        
        # Create and train model
        self.step1_model = MLPRegressor(
            hidden_layer_sizes=(self.config.model_params['hidden_dim'],),
            max_iter=self.config.training_params['num_epochs'],
            learning_rate_init=self.config.training_params['learning_rate'],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        print("Training Step 1 model...")
        self.step1_model.fit(X_train_scaled, y_train_scaled.ravel())
        
        # Evaluate
        y_pred_scaled = self.step1_model.predict(X_train_scaled)
        y_pred = self.step1_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        self.step1_results = {
            'mse': mse,
            'r2': r2,
            'training_loss': self.step1_model.loss_curve_
        }
        
        print(f"Step 1 Results - MSE: {mse:.6f}, R²: {r2:.4f}")
        return self.step1_results
    
    def run_step2_experiment(self) -> Dict[str, Any]:
        """Run Step 2: Parameter prediction experiment
        
        Returns:
            Step 2 results dictionary
        """
        print("\n=== Step 2: Parameter Prediction Experiment ===")
        
        # Prepare training data
        X_train = []  # Flattened sample points
        y_train = []  # True parameters
        
        for exp in self.experiment_data:
            # Flatten x and y samples into feature vector
            features = np.concatenate([exp['x_samples'].flatten(), exp['y_samples'].flatten()])
            X_train.append(features)
            
            # Convert parameters dict to array
            param_array = []
            for key in sorted(exp['true_params'].keys()):
                param_array.append(exp['true_params'][key])
            y_train.append(param_array)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale data
        X_train_scaled = self.step2_scaler_x.fit_transform(X_train)
        y_train_scaled = self.step2_scaler_y.fit_transform(y_train)
        
        # Create and train model
        self.step2_model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=self.config.training_params['num_epochs'],
            learning_rate_init=self.config.training_params['learning_rate'],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        print("Training Step 2 model...")
        self.step2_model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate
        y_pred_scaled = self.step2_model.predict(X_train_scaled)
        y_pred = self.step2_scaler_y.inverse_transform(y_pred_scaled)
        
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        self.step2_results = {
            'mse': mse,
            'r2': r2,
            'training_loss': self.step2_model.loss_curve_,
            'true_params': y_train,
            'predicted_params': y_pred
        }
        
        print(f"Step 2 Results - MSE: {mse:.6f}, R²: {r2:.4f}")
        return self.step2_results
    
    def create_enhanced_visualizations(self):
        """Create comprehensive visualization plots"""
        print("\nGenerating enhanced visualization plots...")
        
        # Set matplotlib to use ASCII-only fonts
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # 1. Training Evolution Comparison
        self._plot_training_evolution()
        
        # 2. Function Fitting Quality Comparison
        self._plot_function_fitting_comparison()
        
        # 3. Parameter Prediction Analysis
        self._plot_parameter_prediction_analysis()
        
        # 4. Original vs Predicted Functions
        self._plot_original_vs_predicted_functions()
        
        # 5. Comprehensive Performance Analysis
        self._plot_comprehensive_performance()
        
        print("All visualization plots saved to 'plots/' directory")
    
    def _plot_training_evolution(self):
        """Plot training loss evolution for both steps"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Step 1 training evolution
        if hasattr(self.step1_model, 'loss_curve_'):
            ax1.plot(self.step1_model.loss_curve_, 'b-', linewidth=2, label='Training Loss')
            ax1.set_title('Step 1: Function Fitting Training Evolution')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Step 2 training evolution
        if hasattr(self.step2_model, 'loss_curve_'):
            ax2.plot(self.step2_model.loss_curve_, 'r-', linewidth=2, label='Training Loss')
            ax2.set_title('Step 2: Parameter Prediction Training Evolution')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_function_fitting_comparison(self):
        """Plot function fitting quality comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Select first 6 experiments for visualization
        for i in range(min(6, len(self.experiment_data))):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            exp = self.experiment_data[i]
            
            # Generate dense x points for smooth curve
            x_dense = np.linspace(
                self.config.data_params['x_range'][0],
                self.config.data_params['x_range'][1],
                200
            )
            
            # True function
            # Update generator with true parameters
            self.generator.coefficients = exp['true_params']
            self.generator.frequency_components = [
                (exp['true_params'][f'a{j}'], exp['true_params'][f'b{j}'], j) 
                for j in range(1, 4)
            ]
            y_true = self.generator.frequency_function(x_dense)
            
            # Predicted function (using Step 1 model)
            if self.step1_model is not None:
                x_dense_scaled = self.step1_scaler_x.transform(x_dense.reshape(-1, 1))
                y_pred_scaled = self.step1_model.predict(x_dense_scaled)
                y_pred = self.step1_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            else:
                y_pred = np.zeros_like(y_true)
            
            # Plot
            ax.plot(x_dense, y_true, 'b-', linewidth=2, label='True Function', alpha=0.8)
            ax.plot(x_dense, y_pred, 'r--', linewidth=2, label='Predicted Function', alpha=0.8)
            ax.scatter(exp['x_samples'], exp['y_samples'], c='green', s=30, alpha=0.6, label='Sample Points')
            
            ax.set_title(f'Experiment {i+1}: Function Fitting')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/function_fitting_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_prediction_analysis(self):
        """Plot parameter prediction accuracy analysis"""
        if 'true_params' not in self.step2_results:
            return
        
        true_params = self.step2_results['true_params']
        pred_params = self.step2_results['predicted_params']
        
        num_params = true_params.shape[1]
        
        fig, axes = plt.subplots(2, (num_params + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if num_params > 1 else [axes]
        
        # Generate parameter names dynamically
        param_names = ['a0']
        for j in range(1, (num_params + 1) // 2 + 1):
            param_names.extend([f'a{j}', f'b{j}'])
        param_names = param_names[:num_params]
        
        for i in range(num_params):
            ax = axes[i]
            
            # Scatter plot: true vs predicted
            ax.scatter(true_params[:, i], pred_params[:, i], alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(true_params[:, i].min(), pred_params[:, i].min())
            max_val = max(true_params[:, i].max(), pred_params[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = r2_score(true_params[:, i], pred_params[:, i])
            
            ax.set_title(f'Parameter {param_names[i]} (R² = {r2:.3f})')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(num_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('plots/parameter_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_original_vs_predicted_functions(self):
        """Plot comparison between original and reconstructed functions from predicted parameters"""
        if 'predicted_params' not in self.step2_results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Select first 6 experiments
        for i in range(min(6, len(self.experiment_data))):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            exp = self.experiment_data[i]
            
            # Generate dense x points
            x_dense = np.linspace(
                self.config.data_params['x_range'][0],
                self.config.data_params['x_range'][1],
                200
            )
            
            # True function
            # Update generator with true parameters
            self.generator.coefficients = exp['true_params']
            self.generator.frequency_components = [
                (exp['true_params'][f'a{j}'], exp['true_params'][f'b{j}'], j) 
                for j in range(1, 4)
            ]
            y_true = self.generator.frequency_function(x_dense)
            
            # Reconstructed function from predicted parameters
            pred_params_array = self.step2_results['predicted_params'][i]
            # Generate parameter names dynamically
            param_names = ['a0']
            for k in range(1, (len(pred_params_array) + 1) // 2 + 1):
                param_names.extend([f'a{k}', f'b{k}'])
            param_names = param_names[:len(pred_params_array)]
            pred_params_dict = {param_names[j]: pred_params_array[j] 
                              for j in range(len(pred_params_array))}
            
            # Update generator with predicted parameters
            self.generator.coefficients = pred_params_dict
            self.generator.frequency_components = [
                (pred_params_dict.get(f'a{j}', 0), pred_params_dict.get(f'b{j}', 0), j) 
                for j in range(1, 4)
            ]
            y_reconstructed = self.generator.frequency_function(x_dense)
            
            # Plot
            ax.plot(x_dense, y_true, 'b-', linewidth=2, label='Original Function', alpha=0.8)
            ax.plot(x_dense, y_reconstructed, 'r--', linewidth=2, label='Reconstructed Function', alpha=0.8)
            ax.scatter(exp['x_samples'], exp['y_samples'], c='green', s=30, alpha=0.6, label='Sample Points')
            
            # Calculate reconstruction error
            mse = mean_squared_error(y_true, y_reconstructed)
            
            ax.set_title(f'Experiment {i+1}: Reconstruction (MSE = {mse:.3f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/original_vs_predicted_functions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_performance(self):
        """Plot comprehensive performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. MSE Comparison
        methods = ['Step 1\n(Function Fitting)', 'Step 2\n(Parameter Prediction)']
        mse_values = [self.step1_results['mse'], self.step2_results['mse']]
        
        bars1 = ax1.bar(methods, mse_values, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax1.set_title('Mean Squared Error Comparison')
        ax1.set_ylabel('MSE')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 2. R² Comparison
        r2_values = [self.step1_results['r2'], self.step2_results['r2']]
        
        bars2 = ax2.bar(methods, r2_values, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax2.set_title('R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars2, r2_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 3. Training Loss Convergence
        if hasattr(self.step1_model, 'loss_curve_') and hasattr(self.step2_model, 'loss_curve_'):
            epochs1 = range(1, len(self.step1_model.loss_curve_) + 1)
            epochs2 = range(1, len(self.step2_model.loss_curve_) + 1)
            
            ax3.plot(epochs1, self.step1_model.loss_curve_, 'b-', linewidth=2, label='Step 1')
            ax3.plot(epochs2, self.step2_model.loss_curve_, 'r-', linewidth=2, label='Step 2')
            ax3.set_title('Training Loss Convergence')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Parameter Prediction Accuracy (if available)
        if 'true_params' in self.step2_results:
            true_params = self.step2_results['true_params']
            pred_params = self.step2_results['predicted_params']
            
            # Generate parameter names dynamically
            param_names = ['a0']
            for k in range(1, (true_params.shape[1] + 1) // 2 + 1):
                param_names.extend([f'a{k}', f'b{k}'])
            param_names = param_names[:true_params.shape[1]]
            param_r2_scores = []
            
            for i in range(true_params.shape[1]):
                r2 = r2_score(true_params[:, i], pred_params[:, i])
                param_r2_scores.append(r2)
            
            bars4 = ax4.bar(param_names, param_r2_scores, color='lightgreen', alpha=0.7)
            ax4.set_title('Parameter Prediction Accuracy (R²)')
            ax4.set_ylabel('R² Score')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars4, param_r2_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self) -> str:
        """Save experiment results to JSON file
        
        Returns:
            Path to saved results file
        """
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Prepare results data
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_experiments': len(self.experiment_data),
                'model_params': self.config.model_params,
                'training_params': self.config.training_params,
                'data_params': self.config.data_params
            },
            'step1_results': {
                'mse': float(self.step1_results['mse']),
                'r2': float(self.step1_results['r2'])
            },
            'step2_results': {
                'mse': float(self.step2_results['mse']),
                'r2': float(self.step2_results['r2'])
            }
        }
        
        # Add parameter prediction accuracy if available
        if 'true_params' in self.step2_results:
            true_params = self.step2_results['true_params']
            pred_params = self.step2_results['predicted_params']
            
            param_names = ['a0', 'a1', 'b1', 'a2', 'b2'][:true_params.shape[1]]
            param_accuracies = {}
            
            for i, name in enumerate(param_names):
                r2 = r2_score(true_params[:, i], pred_params[:, i])
                param_accuracies[name] = float(r2)
            
            results['parameter_accuracies'] = param_accuracies
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/complete_experiment_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")
        return filename
    
    def run_complete_experiment(self, num_experiments: int = 800) -> Dict[str, Any]:
        """Run the complete two-step experiment
        
        Args:
            num_experiments: Number of experiments to generate
            
        Returns:
            Complete experiment results
        """
        print("=== Starting Complete Frequency Analysis Experiment ===")
        
        # Generate experiment data
        self.generate_experiment_data(num_experiments)
        
        # Run both steps
        step1_results = self.run_step1_experiment()
        step2_results = self.run_step2_experiment()
        
        # Create visualizations
        self.create_enhanced_visualizations()
        
        # Save results
        results_file = self.save_results()
        
        print("\n=== Experiment Complete ===")
        print(f"Step 1 (Function Fitting) - MSE: {step1_results['mse']:.6f}, R²: {step1_results['r2']:.4f}")
        print(f"Step 2 (Parameter Prediction) - MSE: {step2_results['mse']:.6f}, R²: {step2_results['r2']:.4f}")
        print(f"Results saved to: {results_file}")
        print("Visualization plots saved to 'plots/' directory")
        
        return {
            'step1': step1_results,
            'step2': step2_results,
            'results_file': results_file
        }


def main():
    """Main function to run the complete experiment"""
    # Load configuration
    config = Config()
    
    # Create and run experiment
    experiment = CompleteFrequencyExperiment(config)
    results = experiment.run_complete_experiment(num_experiments=800)
    
    return results


if __name__ == '__main__':
    main()