# Neural Network Condensation Phenomenon Experiment

## Overview

This experiment investigates the condensation phenomenon in neural networks, focusing on how different activation functions affect parameter clustering during training. The study includes a comprehensive analysis of the custom activation function σ(x) = x * tan(x) alongside standard activation functions.

## Key Features

- **Multiple Activation Functions**: Comparative analysis of 6 different activation functions
- **Parameter Similarity Analysis**: Cosine similarity computation and visualization
- **Condensation Quantification**: Metrics to measure parameter clustering
- **Comprehensive Visualization**: Training curves, similarity heatmaps, and evolution plots
- **Automated Experiment Pipeline**: End-to-end experiment execution with result persistence

## Activation Functions Studied

1. **x_tan_x**: σ(x) = x * tan(x) - Custom activation function with unique properties
2. **ReLU**: Rectified Linear Unit - Standard activation function
3. **Tanh**: Hyperbolic Tangent - Classic activation function
4. **Sigmoid**: Sigmoid function - Traditional activation function
5. **Swish**: x * sigmoid(x) - Modern activation function
6. **GELU**: Gaussian Error Linear Unit - State-of-the-art activation function

## Project Structure

```
condense1/
├── README.md                    # This file
├── experiment_report.md         # Detailed experiment results and analysis
├── experiment.py               # Main experiment script
├── config.py                   # Configuration parameters
├── model.py                    # Neural network model definition
├── trainer.py                  # Training logic and monitoring
├── data_generator.py           # Data generation utilities
├── similarity_analyzer.py      # Parameter similarity analysis
├── activation_functions.py     # Custom activation function implementations
└── condense_experiment_*/      # Experiment results directory
    ├── models/                 # Trained model checkpoints
    ├── plots/                  # Generated visualizations
    └── results/                # Experiment results in JSON format
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

### Installation

```bash
# Navigate to the condense1 directory
cd condense1

# Install required packages (if not already installed)
pip install torch numpy matplotlib seaborn scikit-learn tqdm
```

### Running the Experiment

```bash
# Run the complete experiment
python experiment.py
```

The experiment will:
1. Initialize all activation functions
2. Generate training and test data
3. Train models with each activation function
4. Compute parameter similarities throughout training
5. Generate comprehensive visualizations
6. Save results and trained models

### Configuration

Modify `config.py` to adjust experiment parameters:

```python
# Training parameters
--epochs 3000              # Number of training epochs
--lr 0.01                  # Learning rate
--training_size 50         # Training dataset size
--hidden_layers_width 100  # Hidden layer width
--gamma 1.0                # Initialization parameter

# Experiment parameters
--save_epoch 500           # Model saving interval
--plot_epoch 500           # Plotting interval
```

## Results and Analysis

### Generated Outputs

After running the experiment, you'll find:

#### 1. Visualizations (`plots/` directory)
- **Activation Functions**: Comparison plots of all activation functions
- **Training Curves**: Loss evolution for each activation function
- **Function Fitting**: Target function approximation results
- **Similarity Heatmaps**: Initial and final parameter similarity matrices
- **Similarity Evolution**: How parameter similarity changes over time
- **Comparative Analysis**: Cross-activation function performance comparison

#### 2. Models (`models/` directory)
- Trained model checkpoints for each activation function
- Format: PyTorch state dictionaries (.pth files)

#### 3. Results (`results/` directory)
- Complete experiment results in JSON format
- Includes performance metrics, condensation analysis, and configuration

### Key Metrics

- **Condensation Ratio**: Proportion of highly similar parameter pairs
- **Average Similarity**: Mean cosine similarity between parameters
- **Number of Clusters**: Distinct parameter groupings identified
- **Training/Test Loss**: Model performance metrics

## Understanding the Results

### Condensation Phenomenon

The condensation phenomenon refers to the tendency of neural network parameters to cluster or become similar during training. This experiment quantifies this behavior using:

1. **Cosine Similarity**: Measures the angle between parameter vectors
2. **Clustering Analysis**: Identifies groups of similar parameters
3. **Evolution Tracking**: Monitors how similarity changes over time

### Key Findings

- **x_tan_x activation** shows unique condensation behavior with higher clustering tendency
- **Standard activations** (ReLU, Tanh, etc.) exhibit similar condensation patterns
- **Numerical stability** is crucial for custom activation functions
- **Performance vs. condensation** trade-offs exist

## Customization

### Adding New Activation Functions

1. Implement the activation function in `activation_functions.py`:
```python
class YourActivation(nn.Module):
    def forward(self, x):
        return your_function(x)
```

2. Add it to the configuration in `config.py`:
```python
ACTIVATION_FUNCTIONS = [
    'your_activation',
    # ... existing functions
]
```

3. Update the `get_activation_function` method in `activation_functions.py`

### Modifying Analysis

- **Similarity metrics**: Modify `similarity_analyzer.py` to use different similarity measures
- **Clustering algorithms**: Change clustering methods in the analysis pipeline
- **Visualization**: Customize plots in the respective plotting functions

## Troubleshooting

### Common Issues

1. **NaN values during training**
   - Often occurs with x_tan_x due to numerical instability
   - Consider gradient clipping or modified initialization

2. **Memory issues**
   - Reduce batch size or hidden layer width
   - Use gradient checkpointing for large models

3. **Slow training**
   - Reduce number of epochs or training data size
   - Use GPU acceleration if available

### Performance Optimization

- **GPU Usage**: Set `--device cuda` if CUDA is available
- **Batch Processing**: Adjust batch size based on available memory
- **Parallel Processing**: Consider multi-processing for multiple experiments

## Contributing

When contributing to this experiment:

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Include appropriate error handling
4. Update this README if adding new features
5. Test with different configurations to ensure robustness

## References

- Neural Network Condensation Theory
- Activation Function Analysis Literature
- Parameter Similarity Studies
- Deep Learning Optimization Research

## License

This experiment is part of the ML Lab project and follows the same licensing terms.

---

For detailed results and analysis, see `experiment_report.md`.
For questions or issues, please refer to the main project documentation.