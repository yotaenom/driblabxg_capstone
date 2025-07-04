# Enhanced xG Prediction Pipeline — Driblab Capstone

This repository contains a modular and testable pipeline for predicting enhanced expected goals (xG) using football event and tracking data.

Unlike traditional xG models that only consider shot location or body part, this project is designed to incorporate rich context—such as player positions and defensive pressure—by aligning event (shot) and tracking data.

---

## Objectives

- Align event (shot) and tracking data using timestamps and mappings
- Build a modular pipeline that runs end-to-end from raw data to predictions
- Deliver a structured, professional, and reproducible repository

---

## Technology Stack

- Python 3.8+
- pandas — data handling
- numpy — numerical operations
- joblib — saving/loading model
- argparse — command-line interface
- json — parsing `.json` files

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
├── output/             # Prediction results (CSV)
├── src/                # Core pipeline modules
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── data_validation.py
│   └── inference.py
├── predict_xg.py       # Main script to run the pipeline
├── requirements.txt    # List of required Python libraries
└── README.md           # This guide
```

---

## How to Run the Project

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/driblabxg_capstone.git
cd driblabxg_capstone
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data
- Place your shot, tracking, and mapping JSON files in the appropriate folders under `data/shot_pack/` as shown above.
- Place your trained model file (e.g., `.pkl` or `.joblib`) in the `models/` directory.

### 4. Run the prediction pipeline

```bash
python predict_xg.py \
  --shots_path data/shot_pack/shots \
  --tracking_path data/shot_pack/jsonls \
  --mapping_path data/shot_pack/mappings \
  --model_path models/your_model.pkl \
  --output_path output
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

---