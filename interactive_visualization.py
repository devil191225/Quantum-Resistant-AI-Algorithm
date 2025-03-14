
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class KyberVisualizationDashboard:
    def __init__(self, vulnerability_model=None):
        self.fig = plt.figure(figsize=(14, 10))
        self.gs = gridspec.GridSpec(3, 3)
        
        self.params = {
            'n': 512,     
            'q': 3329,    
            'eta': 2,     
            'k': 2        
        }
        
        self.vulnerability_model = vulnerability_model
        self.reference_configs = {
            'Kyber-512': {'n': 512, 'q': 3329, 'eta': 2, 'k': 2},
            'Kyber-768': {'n': 768, 'q': 3329, 'eta': 2, 'k': 3},
            'Kyber-1024': {'n': 1024, 'q': 3329, 'eta': 2, 'k': 4},
            'AI-Optimized': {'n': 1024, 'q': 7681, 'eta': 2, 'k': 3}  
        }
        
        self.setup_plots()
        self.setup_controls()
        self.update_plots()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        
    def setup_plots(self):
        self.fig.suptitle('Quantum-Resistant Algorithm Parameter Analysis', fontsize=16)
        
        self.ax_security = self.fig.add_subplot(self.gs[0, 0])
        self.ax_security.set_title('Security Level')
        self.ax_security.set_ylim(0, 300)
        self.ax_security.set_ylabel('Security (bits)')
        self.security_bars = self.ax_security.bar(['Current'], [0], color='royalblue')
        
        self.ax_keysize = self.fig.add_subplot(self.gs[0, 1])
        self.ax_keysize.set_title('Key Size Comparison')
        self.ax_keysize.set_ylabel('Size (bytes)')
        
        self.ax_perf = self.fig.add_subplot(self.gs[0, 2])
        self.ax_perf.set_title('Performance')
        self.ax_perf.set_ylabel('Time (ms)')
        
        self.ax_vuln = self.fig.add_subplot(self.gs[1, :2])
        self.ax_vuln.set_title('Vulnerability Analysis')
        
        self.ax_lattice = self.fig.add_subplot(self.gs[1, 2], projection='3d')
        self.ax_lattice.set_title('Lattice Structure')
        
        self.ax_radar = self.fig.add_subplot(self.gs[2, 0], polar=True)
        self.ax_radar.set_title('Configuration Comparison')
        
        self.ax_params = self.fig.add_subplot(self.gs[2, 1:])
        self.ax_params.set_title('Parameter Exploration')
        
    def setup_controls(self):
        slider_color = 'lightgoldenrodyellow'
        
        ax_n = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=slider_color)
        self.slider_n = Slider(ax_n, 'n', 256, 1024, valinit=self.params['n'], valstep=[256, 512, 768, 1024])
        self.slider_n.on_changed(self.update_params)
        
        ax_q = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=slider_color)
        self.slider_q = Slider(ax_q, 'q', 2048, 8192, valinit=self.params['q'], valstep=[2048, 3329, 4096, 7681, 8192])
        self.slider_q.on_changed(self.update_params)
        
        ax_eta = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=slider_color)
        self.slider_eta = Slider(ax_eta, 'η', 2, 5, valinit=self.params['eta'], valstep=[2, 3, 4, 5])
        self.slider_eta.on_changed(self.update_params)
        
        ax_k = plt.axes([0.15, 0.0, 0.65, 0.03], facecolor=slider_color)
        self.slider_k = Slider(ax_k, 'k', 2, 4, valinit=self.params['k'], valstep=[2, 3, 4])
        self.slider_k.on_changed(self.update_params)
        
        ax_reset = plt.axes([0.85, 0.025, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset', color=slider_color, hovercolor='0.975')
        self.button_reset.on_clicked(self.reset_params)
        
        ax_radio = plt.axes([0.025, 0.025, 0.1, 0.15], facecolor=slider_color)
        self.radio_presets = RadioButtons(ax_radio, ('Custom', 'Kyber-512', 'Kyber-768', 'Kyber-1024', 'AI-Optimized'))
        self.radio_presets.on_clicked(self.set_preset_config)
    
    def update_params(self, val):
        self.params['n'] = int(self.slider_n.val)
        self.params['q'] = int(self.slider_q.val)
        self.params['eta'] = int(self.slider_eta.val)
        self.params['k'] = int(self.slider_k.val)
        self.radio_presets.set_active(0)  
        self.update_plots()
    
    def reset_params(self, event):
        self.set_preset_config('Kyber-512')
        self.radio_presets.set_active(1)  
    
    def set_preset_config(self, label):
        if label != 'Custom':
            config = self.reference_configs[label]
            self.params['n'] = config['n']
            self.params['q'] = config['q']
            self.params['eta'] = config['eta']
            self.params['k'] = config['k']
            
            self.slider_n.set_val(config['n'])
            self.slider_q.set_val(config['q'])
            self.slider_eta.set_val(config['eta'])
            self.slider_k.set_val(config['k'])
            
        self.update_plots()
    
    def calculate_metrics(self, params):
        n, q, eta, k = params['n'], params['q'], params['eta'], params['k']
        security_bits = min(256, 0.4 * n * k * np.log2(q / eta))
        public_key_size = k * (n * np.log2(q) / 8) + 32
        private_key_size = k * (n * np.log2(q) / 8) + public_key_size + 64
        base_ops = n * k * np.log2(q)
        encryption_time = base_ops * 0.01
        decryption_time = base_ops * 0.008
        
       
        if self.vulnerability_model:
            vulnerabilities = self.vulnerability_model.analyze_parameter_set(params)
        else:
            lattice_vuln = 1.0 / (n * np.log2(q) / 128)
            side_channel_vuln = 0.2 * (eta / q)
            misuse_vuln = 0.1 * (k / 4) * (eta / 5)
            implementation_vuln = 0.15 / (n / 512)
            quantum_vuln = 0.3 / (n * k * np.log2(q) / 1024)
            
            lattice_vuln = min(max(lattice_vuln, 0), 1)
            side_channel_vuln = min(max(side_channel_vuln, 0), 1)
            misuse_vuln = min(max(misuse_vuln, 0), 1)
            implementation_vuln = min(max(implementation_vuln, 0), 1)
            quantum_vuln = min(max(quantum_vuln, 0), 1)
            
            overall_vuln = 0.3 * lattice_vuln + 0.2 * side_channel_vuln + \
                          0.15 * misuse_vuln + 0.15 * implementation_vuln + \
                          0.2 * quantum_vuln
                          
            vulnerabilities = {
                'lattice_vulnerability': lattice_vuln,
                'side_channel_vulnerability': side_channel_vuln,
                'misuse_vulnerability': misuse_vuln,
                'implementation_vulnerability': implementation_vuln,
                'quantum_vulnerability': quantum_vuln,
                'overall_vulnerability': overall_vuln
            }
        
        return {
            'security_bits': security_bits,
            'public_key_size': public_key_size,
            'private_key_size': private_key_size,
            'encryption_time': encryption_time,
            'decryption_time': decryption_time,
            'vulnerabilities': vulnerabilities
        }
    
    def update_plots(self):
        metrics = self.calculate_metrics(self.params)
        
        ref_metrics = {}
        for name, config in self.reference_configs.items():
            ref_metrics[name] = self.calculate_metrics(config)
        
        self.ax_security.clear()
        self.ax_security.set_title('Security Level')
        self.ax_security.set_ylim(0, 300)
        self.ax_security.set_ylabel('Security (bits)')
        
        configs = ['Current'] + list(self.reference_configs.keys())
        security_levels = [metrics['security_bits']] + [ref_metrics[name]['security_bits'] for name in self.reference_configs]
        
        bars = self.ax_security.bar(configs, security_levels)
        bars[0].set_color('royalblue')
        self.ax_security.axhline(y=128, linestyle='--', color='r', alpha=0.7, label='NIST Level 1')
        self.ax_security.axhline(y=192, linestyle='--', color='g', alpha=0.7, label='NIST Level 3')
        self.ax_security.axhline(y=256, linestyle='--', color='b', alpha=0.7, label='NIST Level 5')
        self.ax_security.legend(fontsize='small')
        
        for bar in bars:
            height = bar.get_height()
            self.ax_security.annotate(f'{height:.1f}',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3),
                                     textcoords="offset points",
                                     ha='center', va='bottom', rotation=90, fontsize=8)
        
        self.ax_keysize.clear()
        self.ax_keysize.set_title('Key Size Comparison')
        self.ax_keysize.set_ylabel('Size (bytes)')
        
        pub_key_sizes = [metrics['public_key_size']] + [ref_metrics[name]['public_key_size'] for name in self.reference_configs]
        priv_key_sizes = [metrics['private_key_size']] + [ref_metrics[name]['private_key_size'] for name in self.reference_configs]
        
        x = np.arange(len(configs))
        width = 0.35
        
        self.ax_keysize.bar(x - width/2, pub_key_sizes, width, label='Public Key')
        self.ax_keysize.bar(x + width/2, priv_key_sizes, width, label='Private Key')
        self.ax_keysize.set_xticks(x)
        self.ax_keysize.set_xticklabels(configs, rotation=45, ha='right')
        self.ax_keysize.legend(fontsize='small')
        
        self.ax_perf.clear()
        self.ax_perf.set_title('Performance')
        self.ax_perf.set_ylabel('Time (ms)')
        
        enc_times = [metrics['encryption_time']] + [ref_metrics[name]['encryption_time'] for name in self.reference_configs]
        dec_times = [metrics['decryption_time']] + [ref_metrics[name]['decryption_time'] for name in self.reference_configs]
        
        self.ax_perf.bar(x - width/2, enc_times, width, label='Encryption')
        self.ax_perf.bar(x + width/2, dec_times, width, label='Decryption')
        self.ax_perf.set_xticks(x)
        self.ax_perf.set_xticklabels(configs, rotation=45, ha='right')
        self.ax_perf.legend(fontsize='small')

        self.ax_vuln.clear()
        self.ax_vuln.set_title('Vulnerability Analysis')
        
        vuln_categories = ['Lattice', 'Side-Channel', 'Misuse', 'Implementation', 'Quantum', 'Overall']
        vuln_configs = ['Current'] + list(ref_metrics.keys())
        vuln_data = np.zeros((len(vuln_configs), len(vuln_categories)))
        vuln_data[0, 0] = metrics['vulnerabilities']['lattice_vulnerability']
        vuln_data[0, 1] = metrics['vulnerabilities']['side_channel_vulnerability']
        vuln_data[0, 2] = metrics['vulnerabilities']['misuse_vulnerability']
        vuln_data[0, 3] = metrics['vulnerabilities']['implementation_vulnerability']
        vuln_data[0, 4] = metrics['vulnerabilities']['quantum_vulnerability']
        vuln_data[0, 5] = metrics['vulnerabilities']['overall_vulnerability']

        for i, name in enumerate(ref_metrics.keys()):
            vuln_data[i+1, 0] = ref_metrics[name]['vulnerabilities']['lattice_vulnerability']
            vuln_data[i+1, 1] = ref_metrics[name]['vulnerabilities']['side_channel_vulnerability']
            vuln_data[i+1, 2] = ref_metrics[name]['vulnerabilities']['misuse_vulnerability']
            vuln_data[i+1, 3] = ref_metrics[name]['vulnerabilities']['implementation_vulnerability']
            vuln_data[i+1, 4] = ref_metrics[name]['vulnerabilities']['quantum_vulnerability']
            vuln_data[i+1, 5] = ref_metrics[name]['vulnerabilities']['overall_vulnerability']
        
        im = self.ax_vuln.imshow(vuln_data, cmap='YlOrRd')
        cbar = plt.colorbar(im, ax=self.ax_vuln)
        cbar.set_label('Vulnerability Score (Lower is Better)')
        self.ax_vuln.set_xticks(np.arange(len(vuln_categories)))
        self.ax_vuln.set_yticks(np.arange(len(vuln_configs)))
        self.ax_vuln.set_xticklabels(vuln_categories)
        self.ax_vuln.set_yticklabels(vuln_configs)
        plt.setp(self.ax_vuln.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(vuln_configs)):
            for j in range(len(vuln_categories)):
                text_color = 'white' if vuln_data[i, j] > 0.5 else 'black'
                self.ax_vuln.text(j, i, f"{vuln_data[i, j]:.2f}",
                                 ha="center", va="center", color=text_color, fontsize=8)
        self.ax_lattice.clear()
        self.ax_lattice.set_title('Lattice Structure')
        n_sample = min(10, self.params['n'] // 50)
        q_sample = self.params['q']
        x = np.linspace(0, n_sample, n_sample)
        y = np.linspace(0, n_sample, n_sample)
        X, Y = np.meshgrid(x, y)
        Z = (X * Y) % q_sample  
        
        self.ax_lattice.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        self.ax_lattice.set_xlabel('Dimension 1')
        self.ax_lattice.set_ylabel('Dimension 2')
        self.ax_lattice.set_zlabel('Value mod q')
        
        self.ax_radar.clear()
        self.ax_radar.set_title('Configuration Comparison')
        categories = ['Security', 'Key Size', 'Encryption', 'Decryption', 'Resistance']
        N = len(categories)
        
        angles = [n/float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] 
        
        def normalize_for_radar(m):
            security_norm = min(m['security_bits'] / 256, 1.0)
            keysize_norm = 1.0 - min(m['public_key_size'] / 2000, 0.99)  
            enc_norm = 1.0 - min(m['encryption_time'] / 100, 0.99)     
            dec_norm = 1.0 - min(m['decryption_time'] / 80, 0.99)       
            resist_norm = 1.0 - min(m['vulnerabilities']['overall_vulnerability'], 0.99) 
            
            return [security_norm, keysize_norm, enc_norm, dec_norm, resist_norm]
        
        current_radar = normalize_for_radar(metrics)
        kyber512_radar = normalize_for_radar(ref_metrics['Kyber-512'])
        optimal_radar = normalize_for_radar(ref_metrics['AI-Optimized'])
        current_radar += current_radar[:1]
        kyber512_radar += kyber512_radar[:1]
        optimal_radar += optimal_radar[:1]
        self.ax_radar.plot(angles, current_radar, 'o-', linewidth=2, label='Current', color='blue')
        self.ax_radar.plot(angles, kyber512_radar, 'o-', linewidth=2, label='Kyber-512', color='green')
        self.ax_radar.plot(angles, optimal_radar, 'o-', linewidth=2, label='AI-Optimized', color='red')
        self.ax_radar.fill(angles, current_radar, alpha=0.1, color='blue')
        self.ax_radar.fill(angles, kyber512_radar, alpha=0.1, color='green')
        self.ax_radar.fill(angles, optimal_radar, alpha=0.1, color='red')
        self.ax_radar.set_xticks(angles[:-1])
        self.ax_radar.set_xticklabels(categories)
        self.ax_radar.legend(loc='upper right', fontsize='small')
        self.ax_params.clear()
        self.ax_params.set_title('Parameter Space Exploration')
        n_values = np.array([256, 512, 768, 1024])
        k_values = np.array([2, 3, 4])
        q_fixed = self.params['q']
        eta_fixed = self.params['eta']
        security_grid = np.zeros((len(k_values), len(n_values)))
        for i, k in enumerate(k_values):
            for j, n in enumerate(n_values):
                params = {'n': n, 'q': q_fixed, 'eta': eta_fixed, 'k': k}
                metrics = self.calculate_metrics(params)
                security_grid[i, j] = metrics['security_bits']
        im = self.ax_params.imshow(security_grid, cmap='viridis', origin='lower',
                                  extent=[n_values[0], n_values[-1], k_values[0], k_values[-1]],
                                  aspect='auto')
        cbar = plt.colorbar(im, ax=self.ax_params)
        cbar.set_label('Security Level (bits)')
        contour_levels = [128, 192, 256]
        contours = self.ax_params.contour(n_values, k_values, security_grid, levels=contour_levels, colors='white', alpha=0.8)
        self.ax_params.clabel(contours, inline=True, fontsize=8, fmt='%d')
        self.ax_params.plot(self.params['n'], self.params['k'], 'ro', markersize=8, label='Current')
        ai_config = self.reference_configs['AI-Optimized']
        self.ax_params.plot(ai_config['n'], ai_config['k'], 'mo', markersize=8, label='AI-Optimized')
        self.ax_params.set_xlabel('Polynomial Dimension (n)')
        self.ax_params.set_ylabel('Module Rank (k)')
        self.ax_params.legend(fontsize='small')
        
        self.fig.canvas.draw_idle()

def run_dashboard(vulnerability_model=None):
    dashboard = KyberVisualizationDashboard(vulnerability_model)
    plt.show()

if __name__ == "__main__":
    from vulnerability_prediction_model import VulnerabilityPredictionModel
    
    vulnerability_model = VulnerabilityPredictionModel()
    vulnerability_model.train(num_samples=500, epochs=10, batch_size=32)
    run_dashboard(vulnerability_model)