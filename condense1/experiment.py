"""
Neural Network Condensation Phenomenon Experiment
Investigating the effect of different activation functions on initial parameter condensation
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import seaborn as sns

# Set matplotlib to use ASCII only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Import custom modules
from config import get_config, ACTIVATION_FUNCTIONS
from model import CondenseNet
from data_generator import DataGenerator
from trainer import CondenseTrainer
from similarity_analyzer import SimilarityAnalyzer
from activation_functions import plot_activation_functions

class CondenseExperiment:
    """Condensation Phenomenon Experiment Class"""
    
    def __init__(self, config=None):
        """
        Initialize experiment
        
        Args:
            config: Experiment configuration
        """
        self.config = config or get_config()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"condense_experiment_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        self.results_dir = os.path.join(self.experiment_dir, "results")
        self.models_dir = os.path.join(self.experiment_dir, "models")
        
        for dir_path in [self.plots_dir, self.results_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize components
        self.data_generator = DataGenerator(boundary=[-1, 1])
        self.similarity_analyzer = SimilarityAnalyzer()
        
        # Generate data
        self.train_data = self.data_generator.generate_training_data(self.config.training_size)
        self.test_data = self.data_generator.generate_test_data(self.config.test_size)
        
        # Store experiment results
        self.experiment_results = {}
        
        print(f"Experiment initialized, results will be saved to: {self.experiment_dir}")
    
    def run_single_experiment(self, activation_name, gamma=None):
        """
        Run single activation function experiment
        
        Args:
            activation_name (str): Activation function name
            gamma (float): Initialization parameter
        
        Returns:
            dict: Experiment results
        """
        if gamma is None:
            gamma = self.config.gamma
        
        print(f"\n{'='*50}")
        print(f"Running experiment: activation={activation_name}, Gamma={gamma}")
        print(f"{'='*50}")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.rand_seed)
        np.random.seed(self.config.rand_seed)
        
        # Create model
        model = CondenseNet(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            hidden_layers_width=self.config.hidden_layers_width,
            activation=activation_name,
            gamma=gamma
        )
        
        # Create trainer
        trainer = CondenseTrainer(
            model=model,
            train_data=self.train_data,
            test_data=self.test_data,
            config=self.config,
            save_dir=self.results_dir
        )
        
        # Train model
        training_results = trainer.train(self.similarity_analyzer)
        
        # Analyze similarity
        similarity_results = self._analyze_similarity_evolution(
            training_results['similarity_history'], activation_name
        )
        
        # Plot results
        self._plot_experiment_results(trainer, similarity_results, activation_name, gamma)
        
        # Save model
        model_path = os.path.join(self.models_dir, f"model_{activation_name}_gamma{gamma}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Organize results
        experiment_result = {
            'activation': activation_name,
            'gamma': gamma,
            'final_train_loss': training_results['train_losses'][-1],
            'final_test_loss': training_results['test_losses'][-1],
            'similarity_analysis': similarity_results,
            'model_path': model_path,
            'config': {
                'lr': self.config.lr,
                'epochs': self.config.epochs,
                'hidden_width': self.config.hidden_layers_width[0],
                'optimizer': self.config.optimizer
            }
        }
        
        return experiment_result
    
    def run_all_experiments(self):
        """Run experiments for all activation functions"""
        print("Starting condensation phenomenon experiments for all activation functions...")
        
        # First plot activation functions
        self._plot_activation_functions()
        
        # Plot target function
        self._plot_target_function()
        
        # Run experiment for each activation function
        for activation_name in ACTIVATION_FUNCTIONS:
            try:
                result = self.run_single_experiment(activation_name)
                self.experiment_results[activation_name] = result
                print(f"✓ {activation_name} experiment completed")
            except Exception as e:
                print(f"✗ {activation_name} experiment failed: {str(e)}")
                continue
        
        # Generate comparison analysis
        self._generate_comparison_analysis()
        
        # Save complete experiment results
        self._save_experiment_results()
        
        print(f"\nAll experiments completed! Results saved in: {self.experiment_dir}")
    
    def _analyze_similarity_evolution(self, similarity_history, activation_name):
        """Analyze similarity evolution"""
        if not similarity_history:
            return {}
        
        # Calculate statistics for each time point
        evolution_stats = []
        for sim_matrix in similarity_history:
            analysis = self.similarity_analyzer.analyze_condensation(sim_matrix)
            evolution_stats.append(analysis)
        
        # Plot similarity evolution
        self.similarity_analyzer.plot_similarity_evolution(
            similarity_history,
            save_path=os.path.join(self.plots_dir, f"similarity_evolution_{activation_name}.png")
        )
        
        return {
            'initial_analysis': evolution_stats[0] if evolution_stats else {},
            'final_analysis': evolution_stats[-1] if evolution_stats else {},
            'evolution_stats': evolution_stats
        }
    
    def _plot_experiment_results(self, trainer, similarity_results, activation_name, gamma):
        """Plot single experiment results"""
        # Training curves
        trainer.plot_training_curves(
            save_path=os.path.join(self.plots_dir, f"training_curves_{activation_name}.png")
        )
        
        # Function fitting results
        trainer.plot_function_fitting(
            save_path=os.path.join(self.plots_dir, f"function_fitting_{activation_name}.png")
        )
        
        # Similarity heatmaps
        if trainer.similarity_history:
            # Initial similarity
            self.similarity_analyzer.plot_similarity_heatmap(
                trainer.similarity_history[0],
                title=f"{activation_name} - Initial Parameter Similarity",
                save_path=os.path.join(self.plots_dir, f"similarity_initial_{activation_name}.png")
            )
            
            # Final similarity
            self.similarity_analyzer.plot_similarity_heatmap(
                trainer.similarity_history[-1],
                title=f"{activation_name} - Final Parameter Similarity",
                save_path=os.path.join(self.plots_dir, f"similarity_final_{activation_name}.png")
            )
    
    def _plot_activation_functions(self):
        """Plot all activation functions"""
        from activation_functions import plot_activation_functions
        
        # Save current directory, switch to plots directory
        original_dir = os.getcwd()
        os.chdir(self.plots_dir)
        
        try:
            plot_activation_functions()
        finally:
            os.chdir(original_dir)
    
    def _plot_target_function(self):
        """Plot target function"""
        self.data_generator.plot_target_function(
            save_path=os.path.join(self.plots_dir, "target_function.png")
        )
    
    def _generate_comparison_analysis(self):
        """Generate comparison analysis for different activation functions"""
        if not self.experiment_results:
            return
        
        # Extract comparison data
        comparison_data = {}
        for activation, result in self.experiment_results.items():
            if 'similarity_analysis' in result and 'final_analysis' in result['similarity_analysis']:
                final_analysis = result['similarity_analysis']['final_analysis']
                comparison_data[activation] = {
                    'condensation_ratio': final_analysis.get('condensation_ratio', 0),
                    'avg_similarity': final_analysis.get('avg_similarity', 0),
                    'std_similarity': final_analysis.get('std_similarity', 0),
                    'n_clusters': len(final_analysis.get('clusters', [])),
                    'final_train_loss': result['final_train_loss'],
                    'final_test_loss': result['final_test_loss']
                }
        
        # Plot comparison charts
        if comparison_data:
            self.similarity_analyzer.compare_activation_functions(
                comparison_data,
                save_path=os.path.join(self.plots_dir, "activation_comparison.png")
            )
            
            # Plot detailed comparison
            self._plot_detailed_comparison(comparison_data)
    
    def _plot_detailed_comparison(self, comparison_data):
        """Plot detailed comparison analysis charts"""
        activations = list(comparison_data.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = [
            ('condensation_ratio', 'Condensation Ratio'),
            ('avg_similarity', 'Average Similarity'),
            ('std_similarity', 'Similarity Std Dev'),
            ('n_clusters', 'Number of Clusters'),
            ('final_train_loss', 'Final Training Loss'),
            ('final_test_loss', 'Final Test Loss')
        ]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(activations)))
        
        for i, (metric, title) in enumerate(metrics):
            values = [comparison_data[act][metric] for act in activations]
            
            bars = axes[i].bar(activations, values, color=colors)
            axes[i].set_title(title, fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}' if isinstance(value, float) else f'{value}',
                           ha='center', va='bottom', fontsize=10)
            
            # Use log scale for loss
            if 'loss' in metric:
                axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "detailed_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
    
    def _save_experiment_results(self):
        """Save complete experiment results"""
        # Save as JSON format
        results_file = os.path.join(self.results_dir, "complete_experiment_results.json")
        
        # Prepare data to save
        save_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'lr': float(self.config.lr),
                    'epochs': int(self.config.epochs),
                    'training_size': int(self.config.training_size),
                    'test_size': int(self.config.test_size),
                    'hidden_layers_width': [int(w) for w in self.config.hidden_layers_width],
                    'gamma': float(self.config.gamma),
                    'optimizer': str(self.config.optimizer),
                    'device': str(self.config.device)
                }
            },
            'results': {}
        }
        
        # Add results for each activation function
        for activation, result in self.experiment_results.items():
            save_data['results'][activation] = {
                'final_train_loss': float(result['final_train_loss']),
                'final_test_loss': float(result['final_test_loss']),
                'condensation_ratio': float(result['similarity_analysis']['final_analysis'].get('condensation_ratio', 0)),
                'avg_similarity': float(result['similarity_analysis']['final_analysis'].get('avg_similarity', 0)),
                'n_clusters': int(len(result['similarity_analysis']['final_analysis'].get('clusters', []))),
                'config': {
                    'lr': float(result['config']['lr']),
                    'epochs': int(result['config']['epochs']),
                    'hidden_width': int(result['config']['hidden_width']),
                    'optimizer': str(result['config']['optimizer'])
                }
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Complete experiment results saved to: {results_file}")

def main():
    """Main function"""
    print("Neural Network Condensation Phenomenon Experiment")
    print("Investigating the effect of different activation functions on initial parameter condensation")
    print("="*60)
    
    # Get configuration
    config = get_config()
    
    # Modify configuration here if needed
    config.epochs = 3000
    config.lr = 0.01
    config.training_size = 50
    config.hidden_layers_width = [100]
    config.gamma = 1.0
    
    print(f"Experiment Configuration:")
    print(f"  Training epochs: {config.epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Training data size: {config.training_size}")
    print(f"  Hidden layer width: {config.hidden_layers_width}")
    print(f"  Initialization parameter gamma: {config.gamma}")
    print(f"  Activation functions: {ACTIVATION_FUNCTIONS}")
    
    # Create and run experiment
    experiment = CondenseExperiment(config)
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()