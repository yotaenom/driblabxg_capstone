# Model Management

The project includes a model registry system for tracking model versions, metadata, and performance metrics.

## Register an Existing Model

To register your existing `xgboost_original.pkl` model:

```bash
python register_model.py
```

This will:
- Load your existing model
- Register it with metadata
- Create a `metadata.json` file in `models/registry/`

## Register a New Model

```python
from src.model_registry import save_model_with_metadata

# Save and register a new model
model_id = save_model_with_metadata(
    model=your_trained_model,
    filepath="models/registry/xg_model_v2.pkl",
    model_name="xg_predictor",
    version="2.0",
    metrics={
        "accuracy": 0.85,
        "auc": 0.92,
        "features": ["R", "shooting_angle", "PDI_complete", "gk_line_offset"]
    },
    description="Improved xG prediction model with enhanced features"
)
```

## List and Manage Models

```python
from src.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# List all registered models
models = registry.list_models()
for model_id, info in models.items():
    print(f"{model_id}: {info['description']}")

# Get specific model information
model_info = registry.get_model_info("xg_predictor_v1.0")

# Get the latest version of a model
latest_model = registry.get_latest_model("xg_predictor")

# Update model metrics
registry.update_metrics("xg_predictor_v1.0", {"test_accuracy": 0.87})

# Deactivate a model version
registry.deactivate_model("xg_predictor_v0.9")
```

## Model Registry Features

- Version control: Track multiple versions of the same model
- Metadata storage: Store model descriptions, metrics, and file information
- Performance tracking: Record and update model performance metrics
- Model lifecycle: Activate/deactivate model versions
- Automatic discovery: Find the latest version of a model

Registry structure:
```
models/registry/
├── metadata.json          # Model registry database
├── xgboost_original.pkl   # Model file
└── xg_model_v2.pkl        # Additional model versions
```

The `metadata.json` file contains:
- Model IDs and versions
- File paths and sizes
- Creation and registration timestamps
- Performance metrics
- Model descriptions and status 