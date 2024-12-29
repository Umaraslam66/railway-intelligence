# Railway Network Intelligence System 🚂

An AI-powered railway network analysis and optimization system that uses machine learning to detect anomalies, predict delays, and optimize network performance.

## 🎯 Features

- **Network Health Monitoring**: Real-time detection of anomalous track behavior
- **Smart Track Classification**: Automated clustering of similar track segments
- **Delay Prediction**: ML-based forecasting of network delays
- **Interactive Dashboard**: Dynamic visualization of network metrics
- **Performance Analysis**: Comprehensive track performance scoring

## 🔧 Technologies Used

- **Python 3.11+**
- **Machine Learning**: scikit-learn for anomaly detection and clustering
- **Network Analysis**: NetworkX for graph modeling
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly for interactive dashboards

## 📊 ML Models Implemented

1. **Isolation Forest**: Anomaly detection in track behavior
2. **DBSCAN**: Track clustering based on operational characteristics
3. **Linear Regression**: Delay prediction and traffic forecasting

## 🚀 Getting Started

### Prerequisites

```bash
python -m pip install -r requirements.txt



Basic Usage:

from railway_intelligence import RailwaySimulator

# Initialize simulator
simulator = RailwaySimulator(num_stations=15)

# Generate visualization
dashboard = simulator.visualize_network()
dashboard.show()
