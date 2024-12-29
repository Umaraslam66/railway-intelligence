import numpy as np
from sklearn.linear_model import LinearRegression

class TrafficPredictor:
    def __init__(self):
        self.model = LinearRegression()
        
    def predict_traffic(self, G, hours_ahead=24):
        """Predict future traffic using Linear Regression"""
        edge_predictions = {}
        
        for u, v in G.edges():
            X = np.arange(24).reshape(-1, 1)
            y = G[u][v]['historical_delays']
            
            self.model.fit(X, y)
            future_X = np.arange(24, 24 + hours_ahead).reshape(-1, 1)
            predictions = self.model.predict(future_X)
            
            edge_predictions[(u, v)] = predictions.tolist()
            
        return edge_predictions