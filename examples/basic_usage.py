from src.models.network_generator import NetworkGenerator
from src.models.anomaly_detector import AnomalyDetector
from src.models.track_clusterer import TrackClusterer
from src.models.traffic_predictor import TrafficPredictor
from src.visualization.dashboard import RailwayDashboard

def main():
    # Initialize models
    generator = NetworkGenerator(num_stations=15)
    network = generator.create_network()
    train_data = generator.generate_train_data(network)
    
    # Analyze network
    detector = AnomalyDetector()
    clusterer = TrackClusterer()
    predictor = TrafficPredictor()
    
    # Create dashboard
    dashboard = RailwayDashboard(
        network,
        train_data,
        detector,
        clusterer,
        predictor
    )
    
    # Show visualization
    fig = dashboard.create_dashboard()
    fig.show()

if __name__ == "__main__":
    main()