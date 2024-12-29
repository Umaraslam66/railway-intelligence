# setup_project.py
import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    # Define the base structure
    structure = {
        "src": {
            "__init__.py": "",
            "models": {
                "__init__.py": "",
                "network_generator.py": "",
                "anomaly_detector.py": "",
                "track_clusterer.py": "",
                "traffic_predictor.py": ""
            },
            "visualization": {
                "__init__.py": "",
                "dashboard.py": "",
                "plot_components.py": ""
            },
            "utils": {
                "__init__.py": "",
                "data_processor.py": ""
            }
        },
        "tests": {
            "__init__.py": "",
            "test_models.py": "",
            "test_visualization.py": ""
        },
        "examples": {
            "basic_usage.py": ""
        },
        "notebooks": {
            "analysis.ipynb": ""
        },
        "requirements.txt": "numpy\npandas\nnetworkx\nplotly\nscikit-learn",
        "setup.py": "",
        "README.md": "# Railway Network Intelligence\n\nAI-powered railway network analysis and optimization system.",
        ".gitignore": "*.pyc\n__pycache__/\n.ipynb_checkpoints/\n*.egg-info/\ndist/\nbuild/"
    }
    
    # Create the directory structure
    root_dir = Path("railway_intelligence")
    root_dir.mkdir(exist_ok=True)
    
    def create_structure(current_path, structure):
        for name, content in structure.items():
            path = current_path / name
            if isinstance(content, dict):
                path.mkdir(exist_ok=True)
                create_structure(path, content)
            else:
                path.write_text(content)
    
    create_structure(root_dir, structure)
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_directory_structure()