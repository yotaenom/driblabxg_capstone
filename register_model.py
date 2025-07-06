#!/usr/bin/env python3
"""
Script to register the existing xgboost_original.pkl model in the model registry.
"""

import pickle
import os
from src.model_registry import save_model_with_metadata

def register_existing_model():
    """Register the existing xgboost_original.pkl model."""
    
    model_path = "models/registry/xgboost_original.pkl"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load the existing model
        print(f"Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        print(f"✓ Model loaded successfully")
        print(f"Model type: {type(model)}")
        
        # Register the model
        print("Registering model in registry...")
        model_id = save_model_with_metadata(
            model=model,
            filepath=model_path,
            model_name="xg_predictor",
            version="1.0",
            metrics={
                "model_type": str(type(model)),
                "features": ["R", "shooting_angle", "PDI_complete", "gk_line_offset"],
                "description": "Original xG model with tracking features"
            },
            description="XGBoost model trained on tracking data features for xG prediction"
        )
        
        print(f"✓ Model registered successfully!")
        print(f"Model ID: {model_id}")
        
        # Show registry info
        from src.model_registry import ModelRegistry
        registry = ModelRegistry()
        models = registry.list_models()
        
        print(f"\nRegistry contains {len(models)} models:")
        for model_id, info in models.items():
            print(f"  - {model_id}: {info['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return False

if __name__ == "__main__":
    print("Registering existing xgboost_original.pkl model...")
    success = register_existing_model()
    
    if success:
        print("\n🎉 Model registration completed successfully!")
        print("You can now use the model registry to manage your models.")
    else:
        print("\n❌ Model registration failed.") 