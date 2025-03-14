import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
SECURITY_LEVEL = {
    'NIST_1': 128,  # Level 1 (128 bits)
    'NIST_3': 192,  # Level 3 (192 bits)
    'NIST_5': 256   # Level 5 (256 bits)
}

class KyberParameterOptimizer:
    def __init__(self, target_security_level=SECURITY_LEVEL['NIST_3'], population_size=50, generations=30):
        self.target_security_level = target_security_level
        self.population_size = population_size
        self.generations = generations
        self.param_ranges = {
            'n': [256, 512, 768, 1024],       
            'q': [2048, 3329, 4096, 7681, 8192],  
            'eta': [2, 3, 4, 5],              
            'k': [2, 3, 4]                    
        }
        self.standard_configs = {
            'kyber_512': {
                'params': {'n': 512, 'q': 3329, 'eta': 2, 'k': 2},
                'metrics': None  
            },
            'kyber_768': {
                'params': {'n': 768, 'q': 3329, 'eta': 2, 'k': 3},
                'metrics': None  
            },
            'kyber_1024': {
                'params': {'n': 1024, 'q': 3329, 'eta': 2, 'k': 4},
                'metrics': None  
            }
        }
        
        self._setup_genetic_algorithm()
    
    def _setup_genetic_algorithm(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("n", random.choice, self.param_ranges['n'])
        self.toolbox.register("q", random.choice, self.param_ranges['q'])
        self.toolbox.register("eta", random.choice, self.param_ranges['eta'])
        self.toolbox.register("k", random.choice, self.param_ranges['k'])
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.n, self.toolbox.q, self.toolbox.eta, self.toolbox.k), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_parameters)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_parameters, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _evaluate_parameters(self, individual):

        n, q, eta, k = individual
        security_bits = min(256, 0.4 * n * k * np.log2(q / eta))
        security_penalty = 0
        if security_bits < self.target_security_level:
            security_penalty = (self.target_security_level - security_bits) * 10
        public_key_size = k * (n * np.log2(q) / 8) + 32 
        private_key_size = k * (n * np.log2(q) / 8) + public_key_size + 64  
        base_ops = n * k * np.log2(q)
        encryption_time = base_ops * 0.01 
        decryption_time = base_ops * 0.008
        security_score = min(1.0, security_bits / self.target_security_level)
        key_size_score = 1.0 - min(1.0, (public_key_size / 2000))
        speed_score = 1.0 - min(1.0, (encryption_time / 100))    
        score = (0.6 * security_score + 0.2 * key_size_score + 0.2 * speed_score) - (0.1 * security_penalty)
        self.last_metrics = {
            'security_bits': security_bits,
            'public_key_size': public_key_size,
            'private_key_size': private_key_size,
            'encryption_time': encryption_time,
            'decryption_time': decryption_time,
            'score': score
        }
        
        return (score,)
    
    def _mutate_parameters(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                if i == 0:  # n
                    individual[i] = random.choice(self.param_ranges['n'])
                elif i == 1:  # q
                    individual[i] = random.choice(self.param_ranges['q'])
                elif i == 2:  # eta
                    individual[i] = random.choice(self.param_ranges['eta'])
                elif i == 3:  # k
                    individual[i] = random.choice(self.param_ranges['k'])
        
        return (individual,)
    
    def optimize(self):
        pop = self.toolbox.population(n=self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        hof = tools.HallOfFame(1)
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.generations, stats=stats, halloffame=hof, verbose=True)
        best_individual = hof[0]
        n, q, eta, k = best_individual
        self._evaluate_parameters(best_individual)
        best_metrics = self.last_metrics

        standard_configs = {}
        for name, config in self.standard_configs.items():
            params_list = [config['params']['n'], config['params']['q'], 
                           config['params']['eta'], config['params']['k']]
            self._evaluate_parameters(params_list)
            standard_configs[name] = {
                'params': config['params'],
                'metrics': self.last_metrics
            }
    
            return {
            'best_params': {
                'n': n,
                'q': q,
                'eta': eta,
                'k': k
            },
            'best_metrics': best_metrics,
            'standard_configs': standard_configs,
            'logbook': logbook
        }
    
    def visualize_results(self, results):
        
        best_params = results['best_params']
        best_metrics = results['best_metrics']
        standard_configs = results['standard_configs']
        logbook = results['logbook']

        print("\n=== Best Parameters Found ===")
        print(f"n = {best_params['n']}, q = {best_params['q']}, eta = {best_params['eta']}, k = {best_params['k']}")
        print("\n=== Performance Metrics ===")
        print(f"Security Level: {best_metrics['security_bits']:.1f} bits")
        print(f"Public Key Size: {best_metrics['public_key_size']:.1f} bytes")
        print(f"Private Key Size: {best_metrics['private_key_size']:.1f} bytes")
        print(f"Encryption Time: {best_metrics['encryption_time']:.2f} ms")
        print(f"Decryption Time: {best_metrics['decryption_time']:.2f} ms")
        print(f"Overall Fitness Score: {best_metrics['score']:.4f}")
        print("\n=== Comparison with Standard Configurations ===")
        print(f"{'Configuration':<15} {'Security (bits)':<15} {'Public Key (B)':<15} {'Private Key (B)':<15} {'Encrypt (ms)':<15} {'Decrypt (ms)':<15} {'Score':<10}")
        print("-" * 90)
        print(f"{'AI Optimized':<15} {best_metrics['security_bits']:<15.1f} {best_metrics['public_key_size']:<15.1f} {best_metrics['private_key_size']:<15.1f} {best_metrics['encryption_time']:<15.2f} {best_metrics['decryption_time']:<15.2f} {best_metrics['score']:<10.4f}")
        for name, config in standard_configs.items():
            metrics = config['metrics']
            print(f"{name:<15} {metrics['security_bits']:<15.1f} {metrics['public_key_size']:<15.1f} {metrics['private_key_size']:<15.1f} {metrics['encryption_time']:<15.2f} {metrics['decryption_time']:<15.2f} {metrics['score']:<10.4f}")
        
        gen = logbook.select("gen")
        fit_maxs = logbook.select("max")
        fit_avgs = logbook.select("avg")
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_maxs, label="Max Fitness")
        plt.plot(gen, fit_avgs, label="Avg Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Improvement Over Generations")
        plt.grid(True)
        plt.legend()

        labels = ['Security', 'Key Size Efficiency', 'Encryption Speed', 'Decryption Speed', 'Overall Score']

        def normalize_for_radar(metrics):
            return [
                min(metrics['security_bits'] / SECURITY_LEVEL['NIST_5'], 1.0),
                1.0 - min(metrics['public_key_size'] / 4096, 0.99),
                1.0 - min(metrics['encryption_time'] / 100, 0.99),
                1.0 - min(metrics['decryption_time'] / 100, 0.99),
                metrics['score']
            ]
        
        best_radar = normalize_for_radar(best_metrics)
        kyber512_radar = normalize_for_radar(standard_configs['kyber_512']['metrics'])
        kyber1024_radar = normalize_for_radar(standard_configs['kyber_1024']['metrics'])
        
        N = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        best_radar += best_radar[:1]
        kyber512_radar += kyber512_radar[:1]
        kyber1024_radar += kyber1024_radar[:1]
        
        ax.plot(angles, best_radar, 'o-', linewidth=2, label='AI Optimized')
        ax.plot(angles, kyber512_radar, 'o-', linewidth=2, label='Kyber-512')
        ax.plot(angles, kyber1024_radar, 'o-', linewidth=2, label='Kyber-1024')
        ax.fill(angles, best_radar, alpha=0.1)
        ax.fill(angles, kyber512_radar, alpha=0.1)
        ax.fill(angles, kyber1024_radar, alpha=0.1)
        plt.xticks(angles[:-1], labels)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    optimizer = KyberParameterOptimizer(target_security_level=SECURITY_LEVEL['NIST_3'])
    results = optimizer.optimize()
    optimizer.visualize_results(results)