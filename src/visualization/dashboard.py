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

