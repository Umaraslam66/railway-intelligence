from sklearn.cluster import DBSCAN

class TrackClusterer:
    def __init__(self, eps=0.3, min_samples=2):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        
    def cluster_tracks(self, G):
        """Cluster similar tracks using DBSCAN"""
        track_features = []
        for u, v in G.edges():
            features = [
                G[u][v]['speed_limit'] / 300,
                G[u][v]['maintenance_score'],
                G[u][v]['energy_efficiency']
            ]
            track_features.append(features)
            
        clusters = self.model.fit_predict(track_features)
        return {edge: label for edge, label in zip(G.edges(), clusters)}