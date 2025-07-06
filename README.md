# Enhanced xG Prediction Pipeline — Driblab Capstone

This repository contains a modular and testable pipeline for predicting enhanced expected goals (xG) using football event and tracking data.

Unlike traditional xG models that only consider shot location or body part, this project incorporates rich context—such as player positions and defensive pressure—by aligning event (shot) and tracking data.

---

## Objectives

- Data alignment: Align event (shot) and tracking data using timestamps and mappings
- Modular pipeline: Build an end-to-end pipeline from raw data to predictions
- Professional structure: Deliver a structured, reproducible repository
- MLOps best practices: Implement model management and deployment practices

---

## Technology Stack

- Python 3.8+
- pandas — Data manipulation and analysis
- numpy — Numerical operations
- joblib — Model serialization
- argparse — Command-line interface
- json — Data file parsing
- PyYAML — Configuration management
- scikit-learn — Machine learning framework

---

## MLOps Features

- Configuration management: Centralized `config.yaml` for all pipeline parameters
- Logging system: Comprehensive logging for pipeline execution and debugging
- Model registry: Version control and metadata tracking for models
- CI/CD pipeline: Automated testing across multiple platforms
- Cross-platform compatibility: Tested on macOS, Windows, and Linux

---

## Platform Compatibility

---

## Platform Compatibility

To run this project seamlessly, it should be run on **macOS**. While it is still not tested on **Windows** or **Linux**, the code and dependencies are all cross-platform, so it should run on those systems as well (we’d recommend Ubuntu 24.04 LTS for Linux users). The instructions below provide OS specific commands where necessary. 

---

## How to Run the Project In a Clean Environment

### 1. Clone the Repository

```bash
git clone https://github.com/yotaenom/driblabxg_capstone.git
cd driblabxg_capstone
```

### 2. Create and Activate a Virtual Environment

#### macOS/Linux
```bash
python3 -m venv driblabxg-venv
source driblabxg-venv/bin/activate
```

#### Windows (Command Prompt)
```cmd
python -m venv driblabxg-venv
driblabxg-venv\Scripts\activate
```

#### Windows (PowerShell)
```powershell
python -m venv driblabxg-venv
.\driblabxg-venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Pipeline

Edit `config.yaml` to customize:
- Data paths
- Model settings
- Output preferences
- Validation requirements

### 5. Prepare Your Data
- Place your shot, tracking, and mapping JSON files in the appropriate folders under `data/shot_pack/` as shown below.

---

## Quickstart: Minimal Setup

To run the xG prediction pipeline, you only need the following files and folders:

- `predict_xg.py` — Main pipeline script
- `register_model.py` — Model registration script
- `config.yaml` — Pipeline configuration
- `requirements.txt` — Python dependencies
- `src/` — Core pipeline modules
- `data/` — Your data files (see below)
- `models/` — Model files and registry
- `output/` — Results directory (created automatically)

All other files (notebooks, old CSVs, images, PDFs, etc.) are now in the `historical/` directory for reference and development history.

Data folder structure:
```
data/
└── shot_pack/
    ├── shots/      # Shot data (JSON format)
    ├── jsonls/     # Tracking data (JSONL format)
    └── mappings/   # Mapping files (JSON format)
```

To run the pipeline:
```bash
python predict_xg.py
```

To register a model:
```bash
python register_model.py
```

---

## Folder Structure
```
driblabxg_capstone/
├── data/
│   └── shot_pack/
│       ├── shots/      # Shot data (JSON format)
│       ├── jsonls/     # Tracking data (JSON format)
│       └── mappings/   # Mapping files (JSON format)
├── models/             # Trained model files (.pkl, .joblib, etc.)
│   └── registry/       # Model registry metadata
├── output/             # Prediction results (CSV)
├── src/                # Core pipeline modules
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── data_validation.py
│   ├── inference.py
│   ├── logger.py       # Centralized logging
│   └── model_registry.py # Model versioning
├── historical/         # Development files (notebooks, old data, etc.)
├── .github/workflows/  # CI/CD pipeline
│   └── test.yml
├── config.yaml         # Pipeline configuration
├── predict_xg.py       # Main script to run the pipeline
├── register_model.py   # Model registration script
├── requirements.txt    # List of required Python libraries
└── README.md           # This guide
```

---

## Model Management

The project includes a model registry system for tracking model versions, metadata, and performance metrics.

### Register an Existing Model

To register your existing `xgboost_original.pkl` model:

```bash
python register_model.py
```

This will:
- Load your existing model
- Register it with metadata
- Create a `metadata.json` file in `models/registry/`

### Register a New Model

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

### List and Manage Models

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

### Model Registry Features

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

---

## Authors
Martin Gutierrez  
Diego Lopez  
Alexander Karam  
Yotaro Enomoto  
Africa Bajils  
Alejandro Osto

---

## Troubleshooting

| Problem                        | Solution                                              |
|--------------------------------|-------------------------------------------------------|
| `model.pkl` not found          | Ensure your trained model is in the `models/` folder  |
| `shots/*.json` not found       | Place shot files in `data/shot_pack/shots/`           |
| `jsonls/*.json` not found      | Place tracking files in `data/shot_pack/jsonls/`      |
| `predict_xg.py` fails          | Check that all required arguments are provided         |
| Output file not generated      | Check for errors in the console and data formatting    |
| Configuration errors           | Verify `config.yaml` file format and paths            |

If you are missing files, check the `historical/` directory for old notebooks, CSVs, and documentation from previous development stages.

---