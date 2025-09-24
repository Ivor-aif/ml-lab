"""
Similarity Analyzer
For computing and visualizing cosine similarity of neuron parameters
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set matplotlib to use ASCII only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SimilarityAnalyzer:
    """Neuron Similarity Analyzer"""
    
    def __init__(self):
        """Initialize analyzer"""
        pass
    
    def compute_cosine_similarity(self, weights):
        """
        Compute cosine similarity of weight matrix
        
        Args:
            weights (torch.Tensor): Weight matrix [num_neurons, input_dim]
        
        Returns:
            np.ndarray: Cosine similarity matrix [num_neurons, num_neurons]
        """
        # Convert to numpy array
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(weights)
        
        return similarity_matrix
    
    def compute_parameter_similarity(self, model):
        """
        Compute similarity of first layer parameters
        
        Args:
            model: Neural network model
        
        Returns:
            dict: Dictionary containing weight and bias similarities
        """
        weights, biases, _ = model.get_hidden_weights()
        
        # Calculate weight similarity
        weight_similarity = self.compute_cosine_similarity(weights)
        
        # Calculate bias similarity (if exists)
        bias_similarity = None
        if biases is not None:
            bias_similarity = self.compute_cosine_similarity(biases.unsqueeze(1))
        
        # Calculate extended parameter similarity (weights + biases)
        if biases is not None:
            extended_params = torch.cat([weights, biases.unsqueeze(1)], dim=1)
            extended_similarity = self.compute_cosine_similarity(extended_params)
        else:
            extended_similarity = weight_similarity
        
        return {
            'weight_similarity': weight_similarity,
            'bias_similarity': bias_similarity,
            'extended_similarity': extended_similarity
        }
    
    def plot_similarity_heatmap(self, similarity_matrix, title="Cosine Similarity Heatmap", 
                               save_path=None, figsize=(10, 8)):
        """
        Plot similarity heatmap
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            title (str): Plot title
            save_path (str): Path to save the plot
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   annot=False,  # Don't show values to avoid clutter
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(title, fontsize=14)
        plt.xlabel('Neuron Index', fontsize=12)
        plt.ylabel('Neuron Index', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
        else:
            plt.show()
    
    def analyze_condensation(self, similarity_matrix, threshold=0.9):
        """
        Analyze condensation phenomenon
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            threshold (float): Similarity threshold
        
        Returns:
            dict: Condensation analysis results
        """
        n_neurons = similarity_matrix.shape[0]
        
        # Calculate number of high similarity pairs
        high_similarity_mask = (similarity_matrix > threshold) & (similarity_matrix < 1.0)
        high_similarity_pairs = np.sum(high_similarity_mask) // 2  # Divide by 2 because matrix is symmetric
        
        # Calculate average similarity (excluding diagonal)
        mask = ~np.eye(n_neurons, dtype=bool)
        avg_similarity = np.mean(similarity_matrix[mask])
        
        # Calculate similarity standard deviation
        std_similarity = np.std(similarity_matrix[mask])
        
        # Find condensed neuron clusters
        clusters = self._find_clusters(similarity_matrix, threshold)
        
        return {
            'n_neurons': n_neurons,
            'high_similarity_pairs': high_similarity_pairs,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'clusters': clusters,
            'condensation_ratio': high_similarity_pairs / (n_neurons * (n_neurons - 1) / 2)
        }
    
    def _find_clusters(self, similarity_matrix, threshold=0.9):
        """
        Find clusters of similar neurons
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            threshold (float): Similarity threshold
        
        Returns:
            list: List of clusters, each cluster contains neuron indices
        """
        n_neurons = similarity_matrix.shape[0]
        visited = np.zeros(n_neurons, dtype=bool)
        clusters = []
        
        for i in range(n_neurons):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster = [i]
            visited[i] = True
            
            # Find other neurons similar to current neuron
            for j in range(i + 1, n_neurons):
                if not visited[j] and similarity_matrix[i, j] > threshold:
                    cluster.append(j)
                    visited[j] = True
            
            # Only keep clusters with multiple neurons
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def plot_similarity_evolution(self, similarity_history, save_path=None):
        """
        Plot similarity evolution during training
        
        Args:
            similarity_history (list): Similarity history records
            save_path (str): Save path
        """
        epochs = range(len(similarity_history))
        avg_similarities = [np.mean(sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]) 
                           for sim_matrix in similarity_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, avg_similarities, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Training Epochs', fontsize=12)
        plt.ylabel('Average Cosine Similarity', fontsize=12)
        plt.title('Neuron Parameter Similarity Evolution', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
        else:
            plt.show()
    
    def compare_activation_functions(self, results_dict, save_path=None):
        """
        Compare condensation effects of different activation functions
        
        Args:
            results_dict (dict): Results dictionary for different activation functions
            save_path (str): Save path
        """
        activation_names = list(results_dict.keys())
        condensation_ratios = [results_dict[name]['condensation_ratio'] 
                              for name in activation_names]
        avg_similarities = [results_dict[name]['avg_similarity'] 
                           for name in activation_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Condensation ratio comparison
        bars1 = ax1.bar(activation_names, condensation_ratios, 
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange'])
        ax1.set_ylabel('Condensation Ratio', fontsize=12)
        ax1.set_title('Condensation Ratio by Activation Function', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, ratio in zip(bars1, condensation_ratios):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        # Average similarity comparison
        bars2 = ax2.bar(activation_names, avg_similarities,
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange'])
        ax2.set_ylabel('Average Cosine Similarity', fontsize=12)
        ax2.set_title('Average Similarity by Activation Function', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, similarity in zip(bars2, avg_similarities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{similarity:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
        else:
            plt.show()

def test_similarity_analyzer():
    """Test similarity analyzer"""
    # Create test data
    n_neurons = 20
    input_dim = 5
    
    # Simulate some weight matrices
    weights1 = torch.randn(n_neurons, input_dim)  # Random weights
    weights2 = torch.randn(n_neurons, input_dim)  # Another set of random weights
    
    # Create some similar weights (simulate condensation phenomenon)
    weights2[0] = weights2[1]  # Make first two neurons identical
    weights2[2] = weights2[3] = weights2[4]  # Make neurons 3-5 identical
    
    # Create analyzer
    analyzer = SimilarityAnalyzer()
    
    # Compute similarity
    sim1 = analyzer.compute_cosine_similarity(weights1)
    sim2 = analyzer.compute_cosine_similarity(weights2)
    
    print("Testing similarity analyzer...")
    print(f"Random weights similarity matrix shape: {sim1.shape}")
    print(f"Condensed weights similarity matrix shape: {sim2.shape}")
    
    # Plot heatmaps
    analyzer.plot_similarity_heatmap(sim1, "Random Initialized Weights Similarity")
    analyzer.plot_similarity_heatmap(sim2, "Condensed Weights Similarity")
    
    # Analyze condensation phenomenon
    analysis1 = analyzer.analyze_condensation(sim1)
    analysis2 = analyzer.analyze_condensation(sim2)
    
    print(f"\nRandom weights analysis: {analysis1}")
    print(f"Condensed weights analysis: {analysis2}")

if __name__ == "__main__":
    test_similarity_analyzer()