import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def detect_anomalies(self, G):
        """Detect anomalous traffic patterns"""
        edge_features = []
        for u, v in G.edges():
            features = [
                G[u][v]['capacity'],
                np.mean(G[u][v]['historical_delays']),
                G[u][v]['maintenance_score'],
                G[u][v]['energy_efficiency']
            ]
            edge_features.append(features)
            
        anomalies = self.model.fit_predict(edge_features)
        return {edge: label for edge, label in zip(G.edges(), anomalies)}