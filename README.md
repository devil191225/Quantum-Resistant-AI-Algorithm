# Project Bio


 Please Run  demo.py Code to test all the programs as follows: 

1.  NETWORK TRAFFIC CRYPTOGRAPHIC VULNERABILITY DETECTION DEMONSTRATION
2.  QUANTUM-RESISTANT VULNERABILITY PREDICTION DEMONSTRATION 
3.  INTERACTIVE PARAMETER VISUALIZATION DASHBOARD DEMONSTRATION
4.  INTEGRATED QUANTUM-RESISTANT AI SECURITY SYSTEM DEMONSTRATION 
5.  QUANTUM-RESISTANT AI SECURITY SYSTEM DEMONSTRATION 

## Quantum-Resistant AI Security Algorithms

This project represents the intersection of artificial intelligence and post-quantum cryptography, addressing one of the most significant security challenges on the horizon: the threat quantum computers pose to current encryption standards.

Developed by a team of researchers specializing in both machine learning and cryptography, this toolkit demonstrates how AI techniques can optimize, analyze, and validate quantum-resistant cryptographic algorithms. The project focuses on CRYSTALS-Kyber, a lattice-based encryption scheme recently selected by NIST for standardization in the post-quantum era.

What makes this approach unique is its use of genetic algorithms and neural networks to explore the vast parameter space of lattice-based cryptography, finding configurations that maintain security while improving efficiency. The interactive visualization tools provide researchers and security professionals with intuitive ways to understand the complex trade-offs between security, performance, and vulnerability.

This open-source initiative aims to accelerate the transition to quantum-resistant cryptography by making advanced optimization and analysis tools available to the broader security community. As organizations worldwide prepare for the "harvest now, decrypt later" threat, where adversaries collect encrypted data today to decrypt it once quantum computers become sufficiently powerful, tools like these will be essential for ensuring digital communications remain secure in the quantum computing era.

# Quantum-Resistant AI Security Algorithms

This repository contains AI-driven tools for optimizing, analyzing, and visualizing post-quantum cryptographic algorithms with a specific focus on CRYSTALS-Kyber parameter optimization and vulnerability prediction.

## Overview

The project leverages artificial intelligence techniques to address three critical aspects of post-quantum cryptography:

1. **Parameter Optimization**: Using genetic algorithms to optimize CRYSTALS-Kyber parameters for an ideal balance of security and efficiency.
2. **Vulnerability Prediction**: Neural network models trained to predict potential vulnerabilities in post-quantum algorithms.
3. **Interactive Visualization**: Interactive dashboards for exploring parameter spaces and security trade-offs.

## Requirements

To run this project, you'll need the following dependencies:

```
numpy
pandas
matplotlib
tensorflow>=2.0.0
scikit-learn
deap
joblib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantum-resistant-ai.git
cd quantum-resistant-ai
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
quantum-resistant-ai/
├── kyber_parameter_optimizer.py      # Genetic algorithm for Kyber parameter optimization
├── vulnerability_prediction_model.py  # Neural network for vulnerability prediction
├── kyber_visualization_dashboard.py   # Interactive visualization dashboard
├── system_integration.py              # Complete system integration demonstration
├── examples/                          # Example scripts and notebooks
├── model_checkpoints/                 # Directory for saved models
├── logs/                              # Training logs
└── README.md                          # This file
```

## How to Run

### 1. Parameter Optimization

To run the parameter optimization, execute:

```bash
python kyber_parameter_optimizer.py
```

This will:
- Initialize a genetic algorithm optimizer
- Run optimization to find the best parameter configuration
- Compare the optimized parameters with standard Kyber configurations
- Display performance metrics and visualization

#### Example Output:
```
=== Best Parameters Found ===
n = 1024, q = 7681, eta = 2, k = 3

=== Performance Metrics ===
Security Level: 209.7 bits
Public Key Size: 1568.8 bytes
Private Key Size: 3169.7 bytes
Encryption Time: 81.56 ms
Decryption Time: 65.25 ms
Overall Fitness Score: 0.8743
```

### 2. Vulnerability Prediction

To train and test the vulnerability prediction model:

```bash
python vulnerability_prediction_model.py
```

This will:
- Generate synthetic vulnerability data
- Train a neural network to predict vulnerabilities
- Analyze standard Kyber configurations and compare vulnerability scores
- Visualize the results

#### Custom Parameter Analysis

You can also analyze custom parameter sets:

```python
from vulnerability_prediction_model import VulnerabilityPredictionModel

# Initialize and train model
model = VulnerabilityPredictionModel()
model.train(num_samples=2000, epochs=50)

# Analyze a custom parameter set
custom_params = {
    'n': 1024,
    'q': 7681,
    'eta': 2,
    'k': 3,
    'implementation_noise': 0.05  # Lower noise (better implementation)
}

vulnerabilities = model.analyze_parameter_set(custom_params)
print(f"Overall Vulnerability Score: {vulnerabilities['overall_vulnerability']:.4f}")
```

### 3. Interactive Visualization

To launch the interactive visualization dashboard:

```bash
python kyber_visualization_dashboard.py
```

The dashboard provides:
- Real-time parameter adjustment with sliders
- Security level visualization
- Key size comparison
- Performance metrics
- Vulnerability heatmap
- Lattice structure visualization
- Radar chart for configuration comparison

#### Dashboard Controls:
- Use sliders to adjust parameters (n, q, eta, k)
- Select preset configurations from the radio buttons
- View visualizations that update in real-time as parameters change

### 4. Complete System Integration

To run the complete integrated system:

```bash
python system_integration.py
```

This demonstrates:
- Integration of all components into a unified system
- Simulation of network traffic analysis for cryptographic vulnerabilities
- Automatic parameter optimization based on detected vulnerabilities
- Performance comparison of different configurations

## Advanced Usage

### Cross-Validation with Enhanced Vulnerability Model

The enhanced vulnerability prediction model offers more advanced features:

```python
from enhanced_vulnerability_prediction_model import VulnerabilityPredictionModel

# Create model with deep architecture
model = VulnerabilityPredictionModel(model_architecture="deep")

# Train with cross-validation
model.train(
    num_samples=5000,
    epochs=200,
    batch_size=64,
    data_complexity="complex",
    cross_validation=True,
    n_folds=5
)

# Generate optimal configuration
optimal_config = model.generate_optimal_configuration(target_security_level=192)

# Perform sensitivity analysis
sensitivity_df = model.parameter_sensitivity_analysis(optimal_config['parameters'])
model.plot_sensitivity_analysis(sensitivity_df)
```

### Saving and Loading Models

To save a trained model:

```python
model_path, scaler_path = model.save_model("my_vulnerability_model")
```

To load a previously saved model:

```python
new_model = VulnerabilityPredictionModel()
new_model.load_model("my_vulnerability_model_model.h5", "my_vulnerability_model_scaler.pkl")
```

## Example Scripts

The `examples/` directory contains scripts demonstrating specific use cases:

1. `optimize_for_embedded.py` - Parameter optimization for resource-constrained devices
2. `analyze_standard_configs.py` - Vulnerability analysis of standard Kyber configurations
3. `parameter_exploration.py` - Exploration of parameter space effects on security and performance

## Interpretation of Results

### Security Levels
- **NIST Level 1 (128 bits)**: Equivalent to AES-128 security
- **NIST Level 3 (192 bits)**: Equivalent to AES-192 security
- **NIST Level 5 (256 bits)**: Equivalent to AES-256 security

### Vulnerability Scores
- **0.0-0.25**: Low vulnerability
- **0.25-0.5**: Moderate vulnerability
- **0.5-0.75**: High vulnerability
- **0.75-1.0**: Critical vulnerability

## Notes on the Implementation

- The security models used are simplified approximations for demonstration purposes
- In a real-world application, more sophisticated cryptanalytic models would be used
- The vulnerability prediction is based on synthetic data and should be calibrated with real-world cryptanalysis results
- The genetic algorithm parameters can be tuned for specific optimization goals

## Research Background

This project is based on research into applying machine learning and genetic algorithms to post-quantum cryptography, particularly the CRYSTALS-Kyber algorithm selected by NIST for standardization in the post-quantum era.

The approach demonstrated here can help organizations prepare for the "harvest now, decrypt later" threat, where adversaries collect encrypted data today to decrypt it once quantum computers become sufficiently powerful.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues and Solutions

#### ImportError: No module named 'deap'
The DEAP library is required for the genetic algorithm component. Install it with:
```bash
pip install deap
```

#### ValueError: Model has not been trained yet
Ensure you call the `train()` method before attempting to analyze parameters or make predictions:
```python
model = VulnerabilityPredictionModel()
model.train(num_samples=500, epochs=10)  # Quick training for testing
```

#### MemoryError during training
Reduce the number of samples or batch size:
```python
model.train(num_samples=1000, batch_size=32)  # Lower values to reduce memory usage
```

#### Dashboard not responding
The visualization dashboard is computationally intensive. Close other applications and reduce the complexity:
```python
run_dashboard(reduced_complexity=True)
```

### Performance Optimization

- For faster training on large datasets, enable GPU acceleration through TensorFlow
- Reduce the number of generations for parameter optimization during testing
- Use the `SimpleVulnerabilityModel` class for quick experimentation

## Step-by-Step Testing Guide

### Testing the Parameter Optimizer

1. Run the basic optimizer:
```bash
python kyber_parameter_optimizer.py
```

2. To test with different security targets:
```python
from kyber_parameter_optimizer import KyberParameterOptimizer, SECURITY_LEVEL

# Target NIST Level 1 (128 bits)
optimizer = KyberParameterOptimizer(target_security_level=SECURITY_LEVEL['NIST_1'])
results = optimizer.optimize()
optimizer.visualize_results(results)

# Target NIST Level 5 (256 bits)
optimizer = KyberParameterOptimizer(target_security_level=SECURITY_LEVEL['NIST_5'])
results = optimizer.optimize()
optimizer.visualize_results(results)
```

3. To change optimization parameters:
```python
optimizer = KyberParameterOptimizer(
    target_security_level=SECURITY_LEVEL['NIST_3'],
    population_size=100,  # Larger population
    generations=50        # More generations
)
results = optimizer.optimize()
```

### Testing the Vulnerability Prediction Model

1. Run basic vulnerability analysis:
```bash
python vulnerability_prediction_model.py
```

2. Test with custom configurations:
```python
from vulnerability_prediction_model import VulnerabilityPredictionModel

model = VulnerabilityPredictionModel()
model.train(num_samples=2000, epochs=20)

# Define configurations to compare
configs = {
    'Standard': {'n': 1024, 'q': 3329, 'eta': 2, 'k': 4},
    'Alternative': {'n': 1024, 'q': 7681, 'eta': 3, 'k': 3},
    'Experimental': {'n': 768, 'q': 8192, 'eta': 2, 'k': 4}
}

# Compare configurations
comparison = model.compare_configurations(configs)
print(comparison)
model.plot_vulnerability_comparison(comparison)
```

3. Generate and analyze a synthetic attack scenario:
```python
# Generate a configuration potentially vulnerable to side-channel attacks
vulnerable_config = {
    'n': 512, 
    'q': 3329, 
    'eta': 5,  # Higher noise parameter
    'k': 2,
    'implementation_noise': 0.3  # Poor implementation quality
}

vulnerabilities = model.analyze_parameter_set(vulnerable_config)
print("Vulnerability Analysis:")
for k, v in vulnerabilities.items():
    print(f"  {k}: {v:.4f}")
```

### Testing the Interactive Dashboard

1. Launch the basic dashboard:
```bash
python kyber_visualization_dashboard.py
```

2. Test with a trained vulnerability model:
```python
from vulnerability_prediction_model import VulnerabilityPredictionModel
from kyber_visualization_dashboard import run_dashboard

# Create and train a model
model = VulnerabilityPredictionModel()
model.train(num_samples=500, epochs=10)  # Quick training

# Launch dashboard with the model
run_dashboard(vulnerability_model=model)
```

3. Explore different parameter configurations:
   - Start with a standard configuration (like Kyber-512)
   - Try increasing `n` to 1024 and observe security improvements
   - Change `q` to 7681 and note the effect on key size
   - Adjust `eta` and observe the effect on vulnerability scores

### Testing the Complete System

1. Run a short simulation:
```bash
python system_integration.py
```

2. Customize the simulation duration:
```python
from quantum_resistant_security_system import QuantumResistantSecuritySystem

system = QuantumResistantSecuritySystem()
system.run_simulation(duration_seconds=60)  # Run for 1 minute
```

3. Test the optimized communication demonstration:
```python
system = QuantumResistantSecuritySystem()
system.initialize_components()
performance_data = system.demonstrate_optimized_communication()
```

## Extending the Project

### Adding New Parameter Types

To add new parameters to the optimization process:

1. Modify the `param_ranges` dictionary in `KyberParameterOptimizer.__init__`:
```python
self.param_ranges = {
    'n': [256, 512, 768, 1024],
    'q': [2048, 3329, 4096, 7681, 8192],
    'eta': [2, 3, 4, 5],
    'k': [2, 3, 4],
    'new_param': [value1, value2, value3]  # Add your new parameter
}
```

2. Update the `_evaluate_parameters` method to include the new parameter in fitness calculation

### Creating Custom Visualization Components

To add new visualizations to the dashboard:

1. Add a new method to the `KyberVisualizationDashboard` class
2. Create a new subplot in the `setup_plots` method
3. Update the visualization in the `update_plots` method

Example for adding a new heat map visualization:
```python
def setup_plots(self):
    # Existing code...
    
    # Add new heat map
    self.ax_new_heatmap = self.fig.add_subplot(self.gs[2, 2])
    self.ax_new_heatmap.set_title('Custom Analysis')

def update_plots(self):
    # Existing code...
    
    # Update new heat map
    self.ax_new_heatmap.clear()
    self.ax_new_heatmap.set_title('Custom Analysis')
    
    # Create data for heatmap
    custom_data = np.random.rand(5, 5)
    im = self.ax_new_heatmap.imshow(custom_data, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=self.ax_new_heatmap)
    cbar.set_label('Custom Metric')
```

### Implementing Additional Cryptographic Algorithms

To extend the system with other post-quantum algorithms:

1. Create a new optimizer class modeled after `KyberParameterOptimizer`
2. Implement appropriate parameter ranges and evaluation functions
3. Add algorithm selection to the dashboard

## Performance Testing

To benchmark the performance of different configurations:

```python
from quantum_resistant_security_system import QuantumResistantSecuritySystem

system = QuantumResistantSecuritySystem()
system.initialize_components()

# Perform performance testing
performance_results = []
for n in [512, 768, 1024]:
    for q in [3329, 7681]:
        for k in [2, 3, 4]:
            config = {
                'name': f"Config_n{n}_q{q}_k{k}",
                'params': {'n': n, 'q': q, 'eta': 2, 'k': k}
            }
            result = system.test_configuration_performance(config)
            performance_results.append(result)

# Display results
import pandas as pd
results_df = pd.DataFrame(performance_results)
print(results_df.sort_values('throughput', ascending=False))
```

## Future Development

Planned improvements for future versions:

1. Integration with actual CRYSTALS-Kyber implementation for real-world performance testing
2. Support for additional post-quantum algorithms (NTRU, Saber, etc.)
3. Enhanced machine learning models trained on real cryptanalytic research data
4. Web interface for easier access to the visualization dashboard
5. Export functionality for optimized parameters to configuration files
6. Integration with TLS libraries for protocol-level testing

## Acknowledgments

This project draws inspiration from:
- NIST Post-Quantum Cryptography Standardization Process
- The CRYSTALS-Kyber team's research papers
- Previous work on AI applications in cryptography
- Open-source genetic algorithm and machine learning frameworks

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository or contact the project maintainer at adityasrikar131@gmail.com or ask92@duke.edu.

---

Thank you for using our Quantum-Resistant AI Security Algorithms toolkit. We hope this project helps advance your understanding of post-quantum cryptography and the application of AI techniques to cryptographic security.



