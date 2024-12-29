import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RailwayNetworkSimulator:
    def __init__(self, num_stations=10):
        self.num_stations = num_stations
        self.G = self._create_network()
        self.train_data = self._generate_train_data()
        self.scaler = StandardScaler()
        self._init_ml_models()
        
    def _init_ml_models(self):
        """Initialize ML models"""
        # Linear regression for traffic prediction
        self.traffic_predictor = LinearRegression()
        
        # Anomaly detection model
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def _create_network(self):
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
    
    def _generate_train_data(self):
        """Generate synthetic train movement data"""
        train_data = []
        current_time = datetime.now()
        
        # Generate 24 hours of train movements
        for hour in range(24):
            time = current_time + timedelta(hours=hour)
            
            # Generate random train movements between stations
            for _ in range(random.randint(10, 30)):
                start, end = random.sample(list(self.G.nodes()), 2)
                train_data.append({
                    'time': time,
                    'train_id': f'Train_{len(train_data)}',
                    'from_station': start,
                    'to_station': end,
                    'load': random.randint(50, 100),  # Percentage of capacity
                    'speed': random.randint(60, 200),  # km/h
                    'delay': random.uniform(0, 30)  # minutes
                })
                
        return pd.DataFrame(train_data)
    
    def predict_future_traffic(self, hours_ahead=24):
        """Predict future traffic using Linear Regression"""
        edge_predictions = {}
        
        for u, v in self.G.edges():
            # Use historical delays as features
            X = np.arange(24).reshape(-1, 1)  # Time points
            y = self.G[u][v]['historical_delays']
            
            # Fit and predict
            self.traffic_predictor.fit(X, y)
            future_X = np.arange(24, 24 + hours_ahead).reshape(-1, 1)
            predictions = self.traffic_predictor.predict(future_X)
            
            edge_predictions[(u, v)] = predictions.tolist()
            
        return edge_predictions
    
    def detect_anomalies(self):
        """Detect anomalous traffic patterns"""
        edge_features = []
        for u, v in self.G.edges():
            features = [
                self.G[u][v]['capacity'],
                np.mean(self.G[u][v]['historical_delays']),
                self.G[u][v]['maintenance_score'],
                self.G[u][v]['energy_efficiency']
            ]
            edge_features.append(features)
            
        # Fit and predict anomalies
        anomalies = self.anomaly_detector.fit_predict(edge_features)
        return {edge: label for edge, label in zip(self.G.edges(), anomalies)}
    
    def cluster_similar_tracks(self):
        """Cluster similar tracks using DBSCAN"""
        track_features = []
        for u, v in self.G.edges():
            features = [
                self.G[u][v]['speed_limit'] / 300,  # Normalize speed
                self.G[u][v]['maintenance_score'],
                self.G[u][v]['energy_efficiency']
            ]
            track_features.append(features)
            
        # Cluster tracks
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(track_features)
        return {edge: label for edge, label in zip(self.G.edges(), clustering.labels_)}

    def visualize_network(self):
        """Enhanced network visualization with analysis insights"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Network Health Status</b><br><sup>Critical sections highlighted in red</sup>',
                '<b>Track Classification Map</b><br><sup>Tracks grouped by characteristics</sup>',
                '<b>Critical Track Performance</b><br><sup>5 busiest routes analysis</sup>',
                '<b>24-Hour Delay Forecast</b><br><sup>Expected delays on major routes</sup>'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatterpolar"}, {"type": "scatter"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        pos = nx.get_node_attributes(self.G, 'pos')
        anomalies = self.detect_anomalies()
        clusters = self.cluster_similar_tracks()
        future_traffic = self.predict_future_traffic()
        
        self._add_health_network(fig, pos, anomalies, 1, 1)
        self._add_track_classification(fig, pos, clusters, 1, 2)
        self._add_performance_radar(fig, 2, 1)
        self._add_delay_forecast(fig, future_traffic, 2, 2)
        
        fig.update_layout(
            title={
                'text': '<b>AI-Powered Railway Network Intelligence Dashboard</b>',
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            showlegend=True,
            width=1500,
            height=1100,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )
        
        # Remove axes for network plots
        for row, col in [(1,1), (1,2)]:
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
        
        # Add timestamp and analysis summary
        fig.add_annotation(
            text=f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            xref="paper", yref="paper",
            x=0.01, y=-0.18,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
        
        return fig

    def _add_health_network(self, fig, pos, anomalies, row, col):
        """Network health visualization with permanent labels"""
        # Add network stats annotation
        anomalous_count = sum(1 for status in anomalies.values() if status == -1)
        stats_text = (
            f"Network Stats:<br>"
            f"Total Tracks: {len(self.G.edges())}<br>"
            f"Critical Sections: {anomalous_count}<br>"
            f"Stations: {len(self.G.nodes())}"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref="x domain", yref="y domain",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            align="left",
            row=row, col=col
        )
        
        # Separate and add tracks
        normal_tracks = [(u,v) for (u,v), status in anomalies.items() if status != -1]
        anomalous_tracks = [(u,v) for (u,v), status in anomalies.items() if status == -1]
        
        # Add tracks with permanent labels for critical sections
        self._add_tracks(fig, normal_tracks, pos, 'rgba(0,100,255,0.6)', 'Normal Tracks', 2, row, col)
        self._add_tracks(fig, anomalous_tracks, pos, 'rgba(255,0,0,0.8)', 'Critical Tracks', 3, row, col, True)
        
        # Add stations with labels
        for node in self.G.nodes():
            x, y = pos[node]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers+text',
                    marker=dict(size=12, color='black', symbol='circle'),
                    text=f'S{node}<br>{self.G.nodes[node]["platforms"]} platforms',
                    textposition="top center",
                    name='Station',
                    showlegend=False
                ),
                row=row, col=col
            )

    def _add_tracks(self, fig, track_list, pos, color, name, width, row, col, show_labels=False):
        """Add tracks with optional permanent labels"""
        for u, v in track_list:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    line=dict(width=width, color=color),
                    name=name,
                    showlegend=True if (u,v) == track_list[0] else False
                ),
                row=row, col=col
            )
            
            if show_labels:
                # Add permanent label for critical tracks
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                label_text = (
                    f"Track: S{u}-S{v}<br>"
                    f"Speed: {self.G[u][v]['speed_limit']}km/h<br>"
                    f"Type: {self.G[u][v]['track_type']}"
                )
                
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    text=label_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    font=dict(size=8),
                    bgcolor="white",
                    bordercolor=color,
                    borderwidth=2,
                    row=row, col=col
                )

    def _add_performance_radar(self, fig, row, col):
        """Radar chart with permanent labels"""
        critical_tracks = sorted(
            self.G.edges(data=True),
            key=lambda x: x[2]['capacity'],
            reverse=True
        )[:5]
        
        categories = ['Maintenance', 'Energy<br>Efficiency', 'Speed', 'Reliability', 'Capacity']
        
        # Add score ranges as a separate trace with low opacity
        score_ranges = [80, 80, 80, 80, 80]  # Reference line for "Good" score
        fig.add_trace(
            go.Scatterpolar(
                r=score_ranges,
                theta=categories,
                name='Good Score (80%)',
                fill='toself',
                fillcolor='rgba(0,255,0,0.1)',
                line=dict(color='rgba(0,255,0,0.5)', dash='dot'),
                showlegend=True
            ),
            row=row, col=col
        )
        
        # Add the actual data traces
        for u, v, data in critical_tracks:
            values = [
                data['maintenance_score'] * 100,
                data['energy_efficiency'] * 100,
                data['speed_limit'] / 3,
                (1 - np.mean(data['historical_delays'])) * 100,
                data['capacity']
            ]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    name=f'Track S{u}-S{v}',
                    fill='toself',
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        # Update the polar layout
        fig.update_layout({
            f'polar{row}{col}': dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticktext=['Poor', 'Fair', 'Good'],
                    tickvals=[30, 60, 90],
                    tickfont=dict(size=10),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=10),
                    rotation=90,
                    direction='clockwise',
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                bgcolor='rgba(255,255,255,0.8)'
            )
        })

        return fig

    def _add_delay_forecast(self, fig, future_traffic, row, col):
        """Delay forecast with annotations"""
        selected_tracks = list(future_traffic.items())[:3]
        hours = list(range(24))
        
        for idx, ((u, v), predictions) in enumerate(selected_tracks):
            line = fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=predictions,
                    name=f'Route S{u}-S{v}',
                    mode='lines+text',
                    line=dict(width=3),
                    opacity=0.8,
                    text=[f'{p:.1f}m' if i % 4 == 0 else '' for i, p in enumerate(predictions)]  # Show delay values every 4 hours
                ),
                row=row, col=col
            )
            
            # Add peak delay annotation
            peak_hour = np.argmax(predictions)
            peak_delay = max(predictions)
            fig.add_annotation(
                x=peak_hour,
                y=peak_delay,
                text=f"Peak: {peak_delay:.1f}min<br>Hour: {peak_hour}",
                showarrow=True,
                arrowhead=2,
                row=row, col=col
            )
        
        fig.update_xaxes(
            title_text="Hours Ahead",
            row=row, col=col,
            showgrid=True
        )
        fig.update_yaxes(
            title_text="Predicted Delay (minutes)",
            row=row, col=col,
            showgrid=True
        )

    def _add_track_classification(self, fig, pos, clusters, row, col):
        """Add track clustering visualization"""
        cluster_colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for (u, v), cluster in clusters.items():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            color = cluster_colors[cluster % len(cluster_colors)] if cluster != -1 else 'grey'
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=2, color=color),
                    name=f'Cluster {cluster}' if cluster != -1 else 'Noise',
                    hoverinfo='text',
                    text=[f"Track: {u}-{v}<br>Cluster: {cluster}"]
                ),
                row=row, col=col
            )
# Create simulator and visualize
simulator = RailwayNetworkSimulator(num_stations=15)
fig = simulator.visualize_network()
fig.show()

# Print insights
print("\nNetwork Analysis Insights:")
print("\n1. Anomaly Detection:")
anomalies = simulator.detect_anomalies()
anomalous_tracks = [(u,v) for (u,v), label in anomalies.items() if label == -1]
print(f"Detected {len(anomalous_tracks)} anomalous tracks that need attention")

print("\n2. Track Clustering:")
clusters = simulator.cluster_similar_tracks()
cluster_counts = pd.Series([label for label in clusters.values() if label != -1]).value_counts()
print("Track clusters found:", len(cluster_counts))
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} tracks")

print("\n3. Future Traffic Predictions:")
predictions = simulator.predict_future_traffic(hours_ahead=6)
for (u,v), pred in list(predictions.items())[:3]:
    print(f"Track {u}-{v}: Next 6 hours delay predictions: {[f'{x:.2f}' for x in pred[:6]]}")