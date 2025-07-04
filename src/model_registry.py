import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ModelRegistry:
    """
    Simple model registry for tracking model versions and metadata.
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.metadata_file = os.path.join(registry_path, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata or create new."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"models": {}}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, 
                      model_path: str, 
                      model_name: str,
                      version: str,
                      metrics: Optional[Dict[str, Any]] = None,
                      description: str = "") -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            version: Model version
            metrics: Optional performance metrics
            description: Model description
            
        Returns:
            Model ID
        """
        model_id = f"{model_name}_v{version}"
        
        # Get file info
        file_size = os.path.getsize(model_path)
        created_time = datetime.fromtimestamp(os.path.getctime(model_path))
        
        # Create model entry
        model_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "file_path": model_path,
            "file_size": file_size,
            "created_time": created_time.isoformat(),
            "registered_time": datetime.now().isoformat(),
            "description": description,
            "metrics": metrics or {},
            "status": "active"
        }
        
        # Add to registry
        self.metadata["models"][model_id] = model_entry
        self._save_metadata()
        
        return model_id
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information by ID."""
        return self.metadata["models"].get(model_id)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models."""
        return self.metadata["models"]
    
    def get_latest_model(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        models = self.metadata["models"]
        versions = []
        
        for model_id, info in models.items():
            if info["model_name"] == model_name and info["status"] == "active":
                versions.append((info["version"], model_id))
        
        if versions:
            # Sort by version and return latest
            versions.sort(key=lambda x: x[0])
            return versions[-1][1]
        
        return None
    
    def update_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """Update model metrics."""
        if model_id in self.metadata["models"]:
            self.metadata["models"][model_id]["metrics"].update(metrics)
            self._save_metadata()
    
    def deactivate_model(self, model_id: str):
        """Deactivate a model version."""
        if model_id in self.metadata["models"]:
            self.metadata["models"][model_id]["status"] = "inactive"
            self._save_metadata()


def save_model_with_metadata(model, 
                           filepath: str, 
                           model_name: str,
                           version: str,
                           metrics: Optional[Dict[str, Any]] = None,
                           description: str = "") -> str:
    """
    Save a model and register it in the registry.
    
    Args:
        model: Model object to save
        filepath: Path to save the model
        model_name: Name of the model
        version: Model version
        metrics: Optional performance metrics
        description: Model description
        
    Returns:
        Model ID
    """
    # Save model
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    
    # Register in registry
    registry = ModelRegistry()
    model_id = registry.register_model(
        filepath, model_name, version, metrics, description
    )
    
    return model_id


if __name__ == "__main__":
    # Test the registry
    registry = ModelRegistry()
    print("Model registry initialized successfully") 