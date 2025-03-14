import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ipaddress
import re
import json
import time

class CryptographicVulnerabilityDetector:
   
    
    def __init__(self, model_path=None):
      
        self.scaler = StandardScaler()
        self.feature_names = None
        
        #
        self.vulnerability_categories = [
            'quantum_vulnerable_exchange',  
            'deprecated_algorithms',        
            'weak_parameters',              
            'implementation_flaws',         
            'protocol_downgrades'           
        ]
        
        
        self.vulnerable_ciphers = ['TLS_RSA', 'RC4', '3DES', 'DES', 'NULL', 'EXPORT', 'MD5', 'anon']
        self.safe_tls_versions = ['TLS 1.2', 'TLS 1.3']
        self.quantum_vulnerable_exchanges = ['RSA', 'STATIC_DH', 'STATIC_ECDH']
        self.deprecated_hashes = ['MD5', 'SHA1']
        self.weak_key_sizes = {
            'RSA': 2048,      
            'DSA': 2048,      
            'DH': 2048,      
            'ECDH': 256,      
            'ECDSA': 256      
        }
        
        
        if model_path:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = None
    
    def extract_features(self, handshake_data):
        
        features = pd.DataFrame()
        
        
        features['is_tls_1_0'] = handshake_data['protocol_version'].apply(
            lambda x: 1 if x == 'TLS 1.0' else 0
        )
        features['is_tls_1_1'] = handshake_data['protocol_version'].apply(
            lambda x: 1 if x == 'TLS 1.1' else 0
        )
        features['is_tls_1_2'] = handshake_data['protocol_version'].apply(
            lambda x: 1 if x == 'TLS 1.2' else 0
        )
        features['is_tls_1_3'] = handshake_data['protocol_version'].apply(
            lambda x: 1 if x == 'TLS 1.3' else 0
        )
        features['is_ssl_3_0'] = handshake_data['protocol_version'].apply(
            lambda x: 1 if x == 'SSL 3.0' else 0
        )
        
        s
        for cipher in self.vulnerable_ciphers:
            features[f'has_{cipher.lower()}'] = handshake_data['cipher_suite'].apply(
                lambda x: 1 if cipher in x else 0
            )
        
        
        for kex in self.quantum_vulnerable_exchanges:
            features[f'uses_{kex.lower()}'] = handshake_data['key_exchange'].apply(
                lambda x: 1 if kex in x else 0
            )
        
        
        for hash_alg in self.deprecated_hashes:
            features[f'uses_{hash_alg.lower()}'] = handshake_data['signature_algorithm'].apply(
                lambda x: 1 if hash_alg in str(x) else 0
            )
        
        
        for key_type, min_size in self.weak_key_sizes.items():
            col_name = f'{key_type.lower()}_key_size'
            if col_name in handshake_data.columns:
                features[f'weak_{key_type.lower()}_key'] = handshake_data[col_name].apply(
                    lambda x: 1 if x < min_size and x > 0 else 0
                )
        
        
        if 'cert_validity_days' in handshake_data.columns:
            features['short_cert_validity'] = handshake_data['cert_validity_days'].apply(
                lambda x: 1 if x < 30 else 0
            )
            features['long_cert_validity'] = handshake_data['cert_validity_days'].apply(
                lambda x: 1 if x > 825 else 0  # More than ~2.25 years
            )
        
        
        if 'session_resumed' in handshake_data.columns:
            features['session_resumed'] = handshake_data['session_resumed']
        
        if 'renegotiation_requested' in handshake_data.columns:
            features['renegotiation_requested'] = handshake_data['renegotiation_requested']
        
        return features
    
    def rule_based_analysis(self, handshake_data):
        
    
        results = pd.DataFrame(index=handshake_data.index, columns=self.vulnerability_categories)
        results = results.fillna(0)
        
        
        results['quantum_vulnerable_exchange'] = handshake_data['key_exchange'].apply(
            lambda x: 1 if any(kex in str(x) for kex in self.quantum_vulnerable_exchanges) else 0
        )
        
        
        cipher_check = handshake_data['cipher_suite'].apply(
            lambda x: 1 if any(cipher in str(x) for cipher in self.vulnerable_ciphers) else 0
        )
        
        hash_check = handshake_data['signature_algorithm'].apply(
            lambda x: 1 if any(hash_alg in str(x) for hash_alg in self.deprecated_hashes) else 0
        )
        
        results['deprecated_algorithms'] = (cipher_check | hash_check).astype(int)
        
        
        weak_params = []
        for key_type, min_size in self.weak_key_sizes.items():
            col_name = f'{key_type.lower()}_key_size'
            if col_name in handshake_data.columns:
                weak_params.append(handshake_data[col_name].apply(
                    lambda x: 1 if (x < min_size and x > 0) else 0
                ))
        
        if weak_params:
            results['weak_parameters'] = pd.concat(weak_params, axis=1).max(axis=1)
        
        
        results['protocol_downgrades'] = handshake_data['protocol_version'].apply(
            lambda x: 1 if x not in self.safe_tls_versions else 0
        )
        
        
        if {'session_resumed', 'renegotiation_requested', 'compression_used'}.issubset(handshake_data.columns):
            
            results['implementation_flaws'] = (
                (handshake_data['session_resumed'] == 1) & 
                (handshake_data['renegotiation_requested'] == 1) &
                (handshake_data['compression_used'] == 1)
            ).astype(int)
        
        return results
    
    def build_model(self, input_shape):
       
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(len(self.vulnerability_categories), activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(self, handshake_data, labels, epochs=40, batch_size=32, validation_split=0.2):
        
        features = self.extract_features(handshake_data)
        self.feature_names = features.columns.tolist()
        
        
        X = self.scaler.fit_transform(features)
        y = labels[self.vulnerability_categories].values
        
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        
        self.model = self.build_model(X_train.shape[1])
        
        
        print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        
        self.evaluate(X_val, y_val)
        
        return history
    
    def analyze_traffic(self, handshake_data):
        
        features = self.extract_features(handshake_data)
        
        results = pd.DataFrame(index=handshake_data.index)
        
        if self.model:
            
            missing_cols = set(self.feature_names) - set(features.columns)
            for col in missing_cols:
                features[col] = 0
                
         
            X = features[self.feature_names]
            
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)
            
            for i, category in enumerate(self.vulnerability_categories):
                results[f"{category}_prob"] = predictions[:, i]
                results[f"{category}_detected"] = (predictions[:, i] > 0.5).astype(int)
            
            
            results['overall_vulnerability'] = (
                0.30 * predictions[:, 0] +  
                0.25 * predictions[:, 1] +  
                0.20 * predictions[:, 2] +  
                0.15 * predictions[:, 3] +  
                0.10 * predictions[:, 4]    
            )
            
            
            results['severity'] = pd.cut(
                results['overall_vulnerability'],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        else:
            
            rule_results = self.rule_based_analysis(handshake_data)
            
            
            for category in self.vulnerability_categories:
                results[f"{category}_detected"] = rule_results[category]
                results[f"{category}_prob"] = rule_results[category].apply(
                    lambda x: 0.75 if x == 1 else 0.25
                )
            
            
            results['overall_vulnerability'] = (
                0.30 * results[f"quantum_vulnerable_exchange_prob"] +
                0.25 * results[f"deprecated_algorithms_prob"] +
                0.20 * results[f"weak_parameters_prob"] +
                0.15 * results[f"implementation_flaws_prob"] +
                0.10 * results[f"protocol_downgrades_prob"]
            )
            
         
            results['severity'] = pd.cut(
                results['overall_vulnerability'],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        return results
    
    def evaluate(self, X_test, y_test):
        
        if self.model is None:
            print("Model has not been trained yet")
            return
        
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        
        for i, category in enumerate(self.vulnerability_categories):
            accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
            tn, fp, fn, tp = confusion_matrix(y_test[:, i], y_pred[:, i]).ravel()
            
           
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            print(f"\n{category}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  False Positive Rate: {fpr:.4f}")
    
    def generate_alerts(self, handshake_data, vulnerability_analysis, threshold=0.5, group_by=None):
        
        alerts = []
        
        
        if group_by and group_by in handshake_data.columns:
            
            groups = handshake_data.groupby(group_by)
            
            for group_value, group_indices in groups.indices.items():
                
                group_analysis = vulnerability_analysis.loc[group_indices]
                
                
                for category in self.vulnerability_categories:
                    prob_col = f"{category}_prob"
                    if (group_analysis[prob_col] > threshold).any():
                        
                        vulnerable_indices = group_analysis[group_analysis[prob_col] > threshold].index
                        vulnerable_connections = handshake_data.loc[vulnerable_indices]
                       
                        max_prob = group_analysis[prob_col].max()
                        alert = {
                            'id': f"{category}_{group_value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            'timestamp': datetime.now().isoformat(),
                            'group_by': group_by,
                            'group_value': group_value,
                            'vulnerability_type': category,
                            'confidence': max_prob,
                            'severity': self._determine_severity(category, max_prob),
                            'affected_connections': len(vulnerable_indices),
                            'protocols': vulnerable_connections['protocol_version'].unique().tolist(),
                            'details': self._generate_alert_details(category, vulnerable_connections),
                            'recommended_actions': self._generate_recommendations(category, vulnerable_connections)
                        }
                        
                        alerts.append(alert)
        else:
            
            for idx, row in vulnerability_analysis.iterrows():
                
                for category in self.vulnerability_categories:
                    prob_col = f"{category}_prob"
                    if row[prob_col] > threshold:
                        connection = handshake_data.loc[idx]
                        
                        alert = {
                            'id': f"{category}_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            'timestamp': datetime.now().isoformat(),
                            'vulnerability_type': category,
                            'confidence': row[prob_col],
                            'severity': self._determine_severity(category, row[prob_col]),
                            'protocol': connection['protocol_version'],
                            'source_ip': connection.get('source_ip', 'unknown'),
                            'destination_ip': connection.get('destination_ip', 'unknown'),
                            'details': self._generate_alert_details(category, connection),
                            'recommended_actions': self._generate_recommendations(category, connection)
                        }
                        
                        alerts.append(alert)
        
        return alerts
    
    def _determine_severity(self, vulnerability_type, confidence):
       
        severity_weights = {
            'quantum_vulnerable_exchange': 0.9,  
            'deprecated_algorithms': 0.8,
            'weak_parameters': 0.7,
            'implementation_flaws': 0.8,
            'protocol_downgrades': 0.6
        }
        
        base_weight = severity_weights.get(vulnerability_type, 0.5)
        weighted_severity = base_weight * confidence
        
        
        if weighted_severity > 0.75:
            return "Critical"
        elif weighted_severity > 0.5:
            return "High"
        elif weighted_severity > 0.25:
            return "Medium"
        else:
            return "Low"
    
    def _generate_alert_details(self, vulnerability_type, connection_data):
        
        if isinstance(connection_data, pd.DataFrame):
            details = f"Detected {vulnerability_type} vulnerability in {len(connection_data)} connections.\n"
            
            if vulnerability_type == 'quantum_vulnerable_exchange':
                key_exchanges = connection_data['key_exchange'].unique()
                details += f"Vulnerable key exchanges: {', '.join(key_exchanges)}\n"
                
            elif vulnerability_type == 'deprecated_algorithms':
                cipher_suites = connection_data['cipher_suite'].unique()
                details += f"Deprecated algorithms: {', '.join(cipher_suites)}\n"
                
            elif vulnerability_type == 'weak_parameters':
                details += "Weak cryptographic parameters detected:\n"
                for key_type, min_size in self.weak_key_sizes.items():
                    col_name = f'{key_type.lower()}_key_size'
                    if col_name in connection_data.columns:
                        weak_keys = connection_data[connection_data[col_name] < min_size]
                        if not weak_keys.empty:
                            details += f"- {len(weak_keys)} connections with weak {key_type} keys (<{min_size} bits)\n"
            
        else:
            details = f"Detected {vulnerability_type} vulnerability.\n"
            
            if vulnerability_type == 'quantum_vulnerable_exchange':
                details += f"Vulnerable key exchange: {connection_data['key_exchange']}\n"
                
            elif vulnerability_type == 'deprecated_algorithms':
                details += f"Deprecated algorithm: {connection_data['cipher_suite']}\n"
                
            elif vulnerability_type == 'weak_parameters':
                details += "Weak cryptographic parameters detected:\n"
                for key_type, min_size in self.weak_key_sizes.items():
                    col_name = f'{key_type.lower()}_key_size'
                    if col_name in connection_data.index and connection_data[col_name] < min_size:
                        details += f"- Weak {key_type} key: {connection_data[col_name]} bits (<{min_size} bits)\n"
        
        return details
    
    def _generate_recommendations(self, vulnerability_type, connection_data):
        recommendations = []
        
        if vulnerability_type == 'quantum_vulnerable_exchange':
            recommendations.append("Replace RSA key exchange with ECDHE or DHE")
            recommendations.append("Prepare for transition to quantum-resistant algorithms like CRYSTALS-Kyber")
            recommendations.append("Consider implementing hybrid key exchange using both classical and post-quantum algorithms")
            
        elif vulnerability_type == 'deprecated_algorithms':
            recommendations.append("Upgrade to TLS 1.2 or TLS 1.3")
            recommendations.append("Use modern cipher suites with AES-GCM or ChaCha20-Poly1305")
            recommendations.append("Replace SHA-1 with SHA-256 or stronger")
            
        elif vulnerability_type == 'weak_parameters':
            recommendations.append("Use RSA keys of at least 2048 bits")
            recommendations.append("Use DH parameters of at least 2048 bits")
            recommendations.append("Use elliptic curves of at least 256 bits (e.g., P-256, X25519)")
            
        elif vulnerability_type == 'implementation_flaws':
            recommendations.append("Update cryptographic libraries to the latest versions")
            recommendations.append("Disable TLS compression")
            recommendations.append("Implement proper certificate validation")
            
        elif vulnerability_type == 'protocol_downgrades':
            recommendations.append("Disable support for SSL 3.0, TLS 1.0, and TLS 1.1")
            recommendations.append("Implement downgrade protection mechanisms like TLS_FALLBACK_SCSV")
            recommendations.append("Configure servers to prefer the strongest protocol version")
        
        return recommendations
    
    def load_test_data(self, file_path):
        data = pd.read_csv(file_path)
        required_columns = ['protocol_version', 'cipher_suite', 'key_exchange']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        return data
    
    def visualize_results(self, analysis_results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        vuln_counts = {}
        for category in self.vulnerability_categories:
            detect_col = f"{category}_detected"
            if detect_col in analysis_results.columns:
                vuln_counts[category] = analysis_results[detect_col].sum()
        ax1 = axes[0, 0]
        ax1.bar(vuln_counts.keys(), vuln_counts.values())
        ax1.set_title('Detected Vulnerabilities by Category')
        ax1.set_xlabel('Vulnerability Category')
        ax1.set_ylabel('Number of Detections')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        if 'severity' in analysis_results.columns:
            ax2 = axes[0, 1]
            severity_counts = analysis_results['severity'].value_counts()
            ax2.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', 
                   colors=['green', 'yellow', 'orange', 'red'])
            ax2.set_title('Alert Severity Distribution')
 
        ax3 = axes[1, 0]
        correlation_data = []
        for cat1 in self.vulnerability_categories:
            for cat2 in self.vulnerability_categories:
                col1 = f"{cat1}_detected"
                col2 = f"{cat2}_detected"
                if col1 in analysis_results.columns and col2 in analysis_results.columns:
                    correlation = np.corrcoef(
                        analysis_results[col1].values,
                        analysis_results[col2].values
                    )[0, 1]
                    correlation_data.append((cat1, cat2, correlation))
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data, columns=['cat1', 'cat2', 'correlation'])
            corr_matrix = corr_df.pivot(index='cat1', columns='cat2', values='correlation')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3)
            ax3.set_title('Vulnerability Correlation Matrix')

        ax4 = axes[1, 1]
        if 'overall_vulnerability' in analysis_results.columns:
            sns.histplot(analysis_results['overall_vulnerability'], kde=True, ax=ax4)
            ax4.set_title('Overall Vulnerability Score Distribution')
            ax4.set_xlabel('Vulnerability Score')
            ax4.set_ylabel('Count')
            
          
            ax4.axvline(x=0.25, color='green', linestyle='--', alpha=0.7)
            ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7)
            ax4.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
            
            
            ax4.text(0.125, ax4.get_ylim()[1]*0.9, 'Low', ha='center', color='green')
            ax4.text(0.375, ax4.get_ylim()[1]*0.9, 'Medium', ha='center', color='orange')
            ax4.text(0.625, ax4.get_ylim()[1]*0.9, 'High', ha='center', color='red')
            ax4.text(0.875, ax4.get_ylim()[1]*0.9, 'Critical', ha='center', color='darkred')
        
        plt.tight_layout()
        plt.show()


def simulate_handshake_data(num_samples=1000):
    
    protocol_versions = ['SSL 3.0', 'TLS 1.0', 'TLS 1.1', 'TLS 1.2', 'TLS 1.3']
    protocol_weights = [0.05, 0.1, 0.15, 0.5, 0.2]  
    
   
    cipher_suites = [
        'TLS_RSA_WITH_AES_128_CBC_SHA',
        'TLS_RSA_WITH_AES_256_CBC_SHA',
        'TLS_RSA_WITH_3DES_EDE_CBC_SHA',
        'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
        'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384',
        'TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256',
        'TLS_AES_128_GCM_SHA256',
        'TLS_AES_256_GCM_SHA384',
        'TLS_CHACHA20_POLY1305_SHA256',
        'TLS_RSA_WITH_RC4_128_SHA'
    ]
    cipher_weights = [0.15, 0.15, 0.05, 0.2, 0.15, 0.1, 0.1, 0.05, 0.03, 0.02]
    
    key_exchanges = ['RSA', 'DHE', 'ECDHE', 'STATIC_DH', 'STATIC_ECDH']
    kex_weights = [0.25, 0.1, 0.5, 0.1, 0.05]
    
    sig_algs = ['RSA-SHA1', 'RSA-SHA256', 'ECDSA-SHA256', 'RSA-PSS-SHA256', 'ED25519']
    sig_weights = [0.1, 0.4, 0.3, 0.1, 0.1]
   
    def random_ip():
        return f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
 
    data = []
    for _ in range(num_samples):
        
        protocol = np.random.choice(protocol_versions, p=protocol_weights)
        
        
        if protocol == 'TLS 1.3':
            cipher = np.random.choice(['TLS_AES_128_GCM_SHA256', 'TLS_AES_256_GCM_SHA384', 'TLS_CHACHA20_POLY1305_SHA256'])
            key_exch = 'ECDHE'  # TLS 1.3 always uses ECDHE or DHE
        else:
            cipher = np.random.choice(cipher_suites, p=cipher_weights)
            key_exch = np.random.choice(key_exchanges, p=kex_weights)
        
        rsa_key_size = np.random.choice([1024, 2048, 3072, 4096], p=[0.2, 0.5, 0.2, 0.1])
        dh_key_size = np.random.choice([768, 1024, 2048, 3072], p=[0.1, 0.3, 0.5, 0.1])
        ecdh_curve_size = np.random.choice([192, 256, 384, 521], p=[0.1, 0.6, 0.2, 0.1])
        
        sig_alg = np.random.choice(sig_algs, p=sig_weights)
        cert_valid_days = np.random.choice([30, 90, 365, 730, 825], p=[0.05, 0.1, 0.6, 0.2, 0.05])
        session_resumed = np.random.choice([0, 1], p=[0.7, 0.3])
        renegotiation = np.random.choice([0, 1], p=[0.95, 0.05])
        compression = np.random.choice([0, 1], p=[0.9, 0.1])
        
        record = {
            'protocol_version': protocol,
            'cipher_suite': cipher,
            'key_exchange': key_exch,
            'signature_algorithm': sig_alg,
            'rsa_key_size': rsa_key_size if 'RSA' in key_exch or 'RSA' in sig_alg else 0,
            'dh_param_size': dh_key_size if 'DH' in key_exch else 0,
            'ecdh_curve_size': ecdh_curve_size if 'ECDH' in key_exch else 0,
            'cert_validity_days': cert_valid_days,
            'session_resumed': session_resumed,
            'renegotiation_requested': renegotiation,
            'compression_used': compression,
            'source_ip': random_ip(),
            'destination_ip': random_ip(),
            'timestamp': datetime.now().isoformat()
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def simulate_quantum_attack(encryption_params, quantum_power=4000):
    
    if encryption_params['algorithm'] in ['RSA', 'ECC']:
        key_bits = encryption_params['key_size']
        operations = 2 * (key_bits ** 3)
        return operations / (quantum_power * 3600)
    
    elif encryption_params['algorithm'] in ['AES', 'ChaCha20']:
        key_bits = encryption_params['key_size']
        operations = 2 ** (key_bits/2)
        return operations / (quantum_power * 3600)
    
    elif encryption_params['algorithm'] == 'Kyber':
        n = encryption_params['n']
        q = encryption_params['q']
        k = encryption_params['k']
        security_bits = 0.4 * n * k * np.log2(q / encryption_params['eta'])
        operations = 2 ** (security_bits * 0.7)  
        return operations / (quantum_power * 3600)

def run_demonstration():
    print("Quantum-Resistant Cryptographic Vulnerability Detection System")
    print("=" * 70)
    
    print("\nGenerating synthetic handshake data...")
    handshake_data = simulate_handshake_data(5000)
    print(f"Generated {len(handshake_data)} handshake records")
    
    detector = CryptographicVulnerabilityDetector()
    
    print("\nGenerating training labels using rule-based analysis...")
    labels = detector.rule_based_analysis(handshake_data)
    
    print("\nTraining the vulnerability detection model...")
    history = detector.train(handshake_data, labels, epochs=10, batch_size=32)
    
    print("\nGenerating test data...")
    test_data = simulate_handshake_data(1000)
    print(f"Generated {len(test_data)} test records")
    
    print("\nAnalyzing test data with trained model...")
    results = detector.analyze_traffic(test_data)
    
    print("\nGenerating alerts...")
    alerts = detector.generate_alerts(test_data, results, threshold=0.7)
    print(f"Generated {len(alerts)} alerts")
    
    if alerts:
        print("\nSample alert:")
        print(json.dumps(alerts[0], indent=2))
    
    print("\nVisualizing results...")
    detector.visualize_results(results)
    
    print("\nComparative Analysis:")
    print("\n1. Comparing with static rule-based analysis")
    rule_results = detector.rule_based_analysis(test_data)
    
    ml_detected = {cat: results[f"{cat}_detected"].sum() for cat in detector.vulnerability_categories}
    rule_detected = {cat: rule_results[cat].sum() for cat in detector.vulnerability_categories}
    
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    for cat in detector.vulnerability_categories:
        true_positives[cat] = ((results[f"{cat}_detected"] == 1) & (rule_results[cat] == 1)).sum()
        false_positives[cat] = ((results[f"{cat}_detected"] == 1) & (rule_results[cat] == 0)).sum()
        false_negatives[cat] = ((results[f"{cat}_detected"] == 0) & (rule_results[cat] == 1)).sum()
    
    print(f"{'Category':<30} {'ML Detected':<15} {'Rule Detected':<15} {'True Positives':<15} {'False Positives':<15}")
    print("-" * 90)
    for cat in detector.vulnerability_categories:
        print(f"{cat:<30} {ml_detected[cat]:<15} {rule_detected[cat]:<15} {true_positives[cat]:<15} {false_positives[cat]:<15}")
    
    total_ml = sum(ml_detected.values())
    total_rule = sum(rule_detected.values())
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    
    ml_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    ml_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    ml_f1 = 2 * ml_precision * ml_recall / (ml_precision + ml_recall) if (ml_precision + ml_recall) > 0 else 0
    
    print("\nOverall Performance Metrics:")
    print(f"ML Precision: {ml_precision:.4f}")
    print(f"ML Recall: {ml_recall:.4f}")
    print(f"ML F1 Score: {ml_f1:.4f}")
    
    print("\nDetection Percentages:")
    print(f"Quantum-vulnerable key exchanges: {(results['quantum_vulnerable_exchange_detected'].sum() / len(results) * 100):.1f}%")
    print(f"Deprecated algorithms: {(results['deprecated_algorithms_detected'].sum() / len(results) * 100):.1f}%")
    print(f"Implementation flaws: {(results['implementation_flaws_detected'].sum() / len(results) * 100):.1f}%")
    
    tns = {}
    for cat in detector.vulnerability_categories:
        tns[cat] = ((results[f"{cat}_detected"] == 0) & (rule_results[cat] == 0)).sum()
    
    total_tn = sum(tns.values())
    false_positive_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    print(f"Overall false positive rate: {false_positive_rate:.4f}")
    
    print("\nPerformance benchmarking:")
    benchmark_data = simulate_handshake_data(100)
    
    start_time = time.time()
    detector.analyze_traffic(benchmark_data)
    ml_processing_time = time.time() - start_time
    
    start_time = time.time()
    detector.rule_based_analysis(benchmark_data)
    rule_processing_time = time.time() - start_time
    
    print(f"ML processing time for 100 records: {ml_processing_time:.4f} seconds ({ml_processing_time * 10:.1f} ms per record)")
    print(f"Rule-based processing time for 100 records: {rule_processing_time:.4f} seconds ({rule_processing_time * 10:.1f} ms per record)")
    
    return detector, results, alerts

if __name__ == "__main__":
    detector, results, alerts = run_demonstration()