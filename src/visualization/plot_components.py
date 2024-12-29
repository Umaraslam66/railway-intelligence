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