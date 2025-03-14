
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from parameter_optimizer import KyberParameterOptimizer
from vulnerability_prediction_model import VulnerabilityPredictionModel
from cryptographic_vulnerability_detector import NetworkCryptoDetector
from interactive_visualization import KyberVisualizationDashboard

def run_parameter_optimization_demo():
    print("\n" + "="*80)
    print("QUANTUM-RESISTANT PARAMETER OPTIMIZATION DEMONSTRATION")
    print("="*80)
    
    print("\nThis demonstration shows how genetic algorithms can optimize")
    print("post-quantum cryptographic parameters for security and performance.")
    optimizer = KyberParameterOptimizer()
    print("\nRunning parameter optimization...")
    results = optimizer.optimize()
    optimizer.visualize_results(results)
    
    return results

def run_vulnerability_prediction_demo():
    print("\n" + "="*80)
    print("QUANTUM-RESISTANT VULNERABILITY PREDICTION DEMONSTRATION")
    print("="*80)
    
    print("\nThis demonstration shows how neural networks can predict")
    print("vulnerabilities in post-quantum cryptographic parameter sets.")
    model = VulnerabilityPredictionModel()
    
    print("\nTraining vulnerability prediction model...")
    model.train(num_samples=1000, epochs=20, batch_size=32)
    
    model.plot_training_history()
    standard_configs = {
        'Kyber-512': {'n': 512, 'q': 3329, 'eta': 2, 'k': 2},
        'Kyber-768': {'n': 768, 'q': 3329, 'eta': 2, 'k': 3},
        'Kyber-1024': {'n': 1024, 'q': 3329, 'eta': 2, 'k': 4}
    }
    
    ai_optimized = {'n': 1024, 'q': 7681, 'eta': 2, 'k': 3, 'implementation_noise': 0.05}
    standard_configs['AI-Optimized'] = ai_optimized
    comparison = model.compare_configurations(standard_configs)
    print("\nVulnerability Analysis Results:")
    print(comparison)
    model.plot_vulnerability_comparison(comparison)
    
    return model

def run_network_detection_demo():
    print("\n" + "="*80)
    print("NETWORK TRAFFIC CRYPTOGRAPHIC VULNERABILITY DETECTION DEMONSTRATION")
    print("="*80)
    
    print("\nThis demonstration shows how AI can detect cryptographic")
    print("vulnerabilities in network traffic to prevent quantum-related threats.")
    
    detector = NetworkCryptoDetector()
    print("\nTraining network traffic detector...")
    history = detector.train(epochs=15, batch_size=32)
    detector.plot_training_history(history)
    print("\nAnalyzing test network traffic...")
    test_traffic = detector.generate_synthetic_traffic_data(num_samples=100)
    analysis = detector.analyze_traffic(test_traffic)
    alerts = detector.generate_alert(analysis)
    print(f"\nGenerated {len(alerts)} alerts from {len(test_traffic)} traffic samples")
    if alerts:
        print("\nSample Alert:")
        sample_alert = alerts[0]
        print(f"Severity: {sample_alert['severity']}")
        print("Detected Vulnerabilities:")
        for vuln in sample_alert['detected_vulnerabilities']:
            print(f"  - {vuln['type']} (confidence: {vuln['confidence']:.2f})")
        print("Recommended Actions:")
        for action in sample_alert['recommended_actions']:
            print(f"  - {action}")
    detector.plot_vulnerability_distribution(analysis)
    
    return detector

def run_interactive_dashboard_demo(vulnerability_model=None):
    print("\n" + "="*80)
    print("INTERACTIVE PARAMETER VISUALIZATION DASHBOARD DEMONSTRATION")
    print("="*80)
    
    print("\nThis demonstration shows an interactive dashboard for exploring")
    print("post-quantum cryptographic parameters and their security impacts.")

    from kyber_visualization_dashboard import run_dashboard
    
    print("\nStarting interactive dashboard...")
    print("Use the sliders to adjust parameters and see the impact on security metrics.")
    print("Close the dashboard window when done to continue the demonstration.")
    
    run_dashboard(vulnerability_model)
    
    return True

def run_integrated_system_demo():
    print("\n" + "="*80)
    print("INTEGRATED QUANTUM-RESISTANT AI SECURITY SYSTEM DEMONSTRATION")
    print("="*80)
    
    print("\nThis demonstration shows the complete integrated system with all components")
    print("working together to provide quantum-resistant security capabilities.")
    system = QuantumResistantSecuritySystem()
    system.run_simulation(duration_seconds=45)
    performance_data = system.demonstrate_optimized_communication()
    
    return system

def main():
    parser = argparse.ArgumentParser(description='Quantum-Resistant AI Security Demo')
    parser.add_argument('--optimizer', action='store_true', help='Run parameter optimizer demo')
    parser.add_argument('--vulnerability', action='store_true', help='Run vulnerability prediction demo')
    parser.add_argument('--network', action='store_true', help='Run network detector demo')
    parser.add_argument('--dashboard', action='store_true', help='Run interactive dashboard demo')
    parser.add_argument('--integrated', action='store_true', help='Run integrated system demo')
    parser.add_argument('--all', action='store_true', help='Run all demos')
    
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("\n" + "="*80)
    print("QUANTUM-RESISTANT AI SECURITY SYSTEM DEMONSTRATION")
    print("="*80)
    
    print("\nThis demonstration showcases AI techniques for enhancing post-quantum")
    print("cryptography, including parameter optimization, vulnerability prediction,")
    print("network traffic analysis, and interactive visualization.")
    
    vuln_model = None
    
    if args.all or args.optimizer:
        optimization_results = run_parameter_optimization_demo()
    
    if args.all or args.vulnerability:
        vuln_model = run_vulnerability_prediction_demo()
    
    if args.all or args.network:
        detector = run_network_detection_demo()
    
    if args.all or args.dashboard:
        run_interactive_dashboard_demo(vuln_model)
    
    if args.all or args.integrated:
        system = run_integrated_system_demo()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    
    print("\nThis demonstration showed how AI techniques can enhance post-quantum")
    print("cryptography through parameter optimization, vulnerability prediction,")
    print("and network traffic analysis. These components work together to provide")
    print("comprehensive quantum-resistant security capabilities.")

if __name__ == "__main__":
    main()