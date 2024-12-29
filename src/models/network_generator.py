import networkx as nx
import random
from datetime import datetime, timedelta
import pandas as pd

class NetworkGenerator:
    def __init__(self, num_stations=10):
        self.num_stations = num_stations
        
    def create_network(self):
        """Create enhanced railway network with realistic attributes"""
        G = nx.watts_strogatz_graph(self.num_stations, 3, 0.3)
        
        # Enhanced edge attributes
        for (u, v) in G.edges():
            G[u][v].update({
                'capacity': random.randint(50, 100),
                'speed_limit': random.randint(60, 300),
                'track_type': random.choice(['high_speed', 'regular', 'freight']),
                'electrified': random.choice([True, False]),
                'maintenance_score': random.uniform(0.6, 1.0),
                'historical_delays': [random.uniform(0, 0.3) for _ in range(24)],
                'energy_efficiency': random.uniform(0.7, 0.95)
            })
        
        # Enhanced node attributes
        positions = nx.spring_layout(G)
        for i in G.nodes():
            G.nodes[i].update({
                'name': f'Station_{i}',
                'pos': positions[i],
                'platforms': random.randint(2, 10),
                'daily_passengers': random.randint(1000, 50000),
                'importance_score': random.uniform(0.1, 1.0),
                'connection_types': random.sample(
                    ['local', 'express', 'freight', 'international'],
                    k=random.randint(1, 4)
                )
            })
            
        return G

    def generate_train_data(self, G):
        """Generate synthetic train movement data"""
        train_data = []
        current_time = datetime.now()
        
        for hour in range(24):
            time = current_time + timedelta(hours=hour)
            for _ in range(random.randint(10, 30)):
                start, end = random.sample(list(G.nodes()), 2)
                train_data.append({
                    'time': time,
                    'train_id': f'Train_{len(train_data)}',
                    'from_station': start,
                    'to_station': end,
                    'load': random.randint(50, 100),
                    'speed': random.randint(60, 200),
                    'delay': random.uniform(0, 30)
                })
                
        return pd.DataFrame(train_data)