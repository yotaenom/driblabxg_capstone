# Enhanced xG Prediction Pipeline — Driblab Capstone

This repository contains a modular and testable pipeline for predicting enhanced expected goals (xG) using football event and tracking data.

Unlike traditional xG models that only consider shot location or body part, this project is designed to incorporate rich context—such as player positions and defensive pressure—by aligning event (shot) and tracking data.

---

## Objectives

- Align event (shot) and tracking data using timestamps and mappings
- Build a modular pipeline that runs end-to-end from raw data to predictions
- Deliver a structured, professional, and reproducible repository
- Implement MLOps best practices for model management and deployment

---

## Technology Stack

- Python 3.8+
- pandas — data handling
- numpy — numerical operations
- joblib — saving/loading model
- argparse — command-line interface
- json — parsing `.json` files
- PyYAML — configuration management

---

## MLOps Features

- **Configuration Management**: Centralized config file for all pipeline parameters
- **Logging System**: Comprehensive logging for pipeline execution and debugging
- **Model Registry**: Version control and metadata tracking for models
- **CI/CD Pipeline**: Automated testing across multiple platforms
- **Cross-Platform Compatibility**: Tested on macOS, Windows, and Linux

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
├── .github/workflows/  # CI/CD pipeline
│   └── test.yml
├── config.yaml         # Pipeline configuration
├── predict_xg.py       # Main script to run the pipeline
├── requirements.txt    # List of required Python libraries
└── README.md           # This guide
```

---

## Platform Compatibility

This project is compatible with **macOS**, **Windows**, and **Linux** (Ubuntu 24.04 recommended). The instructions below provide OS-specific commands where necessary. All code and dependencies are cross-platform and have been tested in clean environments on each operating system.

---

## How to Run the Project in a Clean Environment

### 1. Clone the Repository

```bash
# macOS/Linux/Windows (Git Bash or WSL)
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
- Place your shot, tracking, and mapping JSON files in the appropriate folders under `data/shot_pack/` as shown above.
- Place your trained model file (e.g., `.pkl` or `.joblib`) in the `models/` directory.

### 6. Run the Prediction Pipeline

```bash
python predict_xg.py \
  --shots_path data/shot_pack/shots \
  --tracking_path data/shot_pack/jsonls \
  --mapping_path data/shot_pack/mappings \
  --model_path models/your_model.pkl \
  --output_path output
```

---

## Model Management

### Register a New Model

```python
from src.model_registry import save_model_with_metadata

# Save and register model
model_id = save_model_with_metadata(
    model=your_trained_model,
    filepath="models/xg_model_v1.pkl",
    model_name="xg_predictor",
    version="1.0",
    metrics={"accuracy": 0.85, "auc": 0.92},
    description="First version of xG prediction model"
)
```

### List Registered Models

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry()
models = registry.list_models()
print(models)
```

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

---