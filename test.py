import gc
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# GPU setup (fallback to CPU optimizations)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.cuda.empty_cache()
else:
    print("Using CPU with multi-threading optimizations")
    import multiprocessing
    print(f"CPU cores available: {multiprocessing.cpu_count()}")


class Config:
    TRAIN_PATH = "Z:/Machine_Learning/Irrigation_Needed/playground-series-s6e4/train.csv"
    TEST_PATH = "Z:/Machine_Learning/Irrigation_Needed/playground-series-s6e4/test.csv"
    ORIGINAL_PATH = "Z:/Machine_Learning/Irrigation_Needed/playground-series-s6e4/submission1.csv"

    ##Target and Mapping
    
     # ── Target ───────────────────────────────────────────────────────────────
    TARGET         = "Irrigation_Need"
    TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
    INV_MAPPING    = {0: "Low", 1: "Medium", 2: "High"}

    # ── CV / training ────────────────────────────────────────────────────────
    N_FOLDS       = 5  # Reduced for speed, still robust
    RANDOM_SEED   = 42
    PSEUDO_THRESH = 0.90  # Lower threshold for more pseudo-labels

    # ── Domain thresholds ────────────────────────────────────────────────────
    SOIL_THRESH = 25
    RAIN_THRESH = 300
    TEMP_THRESH = 30
    WIND_THRESH = 10

    # ── Feature groups ───────────────────────────────────────────────────────
    CAT_COLS = [
        "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
    ]
    NUM_COLS = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]

    # ── Logit coefficients (fitted on original data) ──────────────────────────
    LOGIT_COEFS = {
        "Low": {
            "intercept": 16.3173, "soil_lt_25": -11.0237, "temp_gt_30": -5.8559,
            "rain_lt_300": -10.8500, "wind_gt_10": -5.8284,
            "Flowering": -5.4155, "Harvest": 5.5073, "Sowing": 5.2299, "Vegetative": -5.4617,
            "Mulch_No": -3.0014, "Mulch_Yes": 2.8613,
        },
        "Medium": {
            "intercept": 4.6524, "soil_lt_25": 0.3290, "temp_gt_30": -0.0204,
            "rain_lt_300": 0.1542, "wind_gt_10": 0.0841,
            "Flowering": 0.3586, "Harvest": -0.1348, "Sowing": -0.3547, "Vegetative": 0.3334,
            "Mulch_No": 0.1883, "Mulch_Yes": 0.0142,
        },
        "High": {
            "intercept": -20.9697, "soil_lt_25": 10.6947, "temp_gt_30": 5.8763,
            "rain_lt_300": 10.6958, "wind_gt_10": 5.7444,
            "Flowering": 5.0569, "Harvest": -5.3725, "Sowing": -4.8752, "Vegetative": 5.1283,
            "Mulch_No": 2.8131, "Mulch_Yes": -2.8755,
        },
    }
    
        # ── XGBoost hyperparameters ───────────────────────────────────────────────
    XGB_PARAMS = dict(
        max_depth=6,  # Slightly deeper for better accuracy
        learning_rate=0.05,  # Faster convergence
        min_child_weight=1,  # More sensitive to small changes
        subsample=0.8,  # Reduced for speed
        colsample_bytree=0.8,  # Reduced for speed
        gamma=1.0,  # Regularization
        reg_alpha=1e-3,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",  # CPU-optimized
        device="cpu",  # Use CPU for reliability
        enable_categorical=True,
        eval_metric="mlogloss",
        seed=42,
        verbosity=0,  # Minimal output
        nthread=-1,  # Use all CPU cores
    )

cfg = Config()
print(" Config ready")