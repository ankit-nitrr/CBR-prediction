"""
Soil Feature Vector (SFV) Preprocessing and Imputation Module.
Handles soil property validation, feature standardization, and missing value imputation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    "LL",          # Liquid Limit (%)
    "PL",          # Plastic Limit (%)
    "PI",          # Plasticity Index (%)
    "fines_pct",    # Fines content < 0.075 mm (%)
    "sand_pct",     # Sand content (%)
    "gravel_pct",   # Gravel content (%)
    "MDD",         # Maximum Dry Density (g/cm3)
    "OMC"          # Optimum Moisture Content (%)
]

TARGET_COL = "CBR"

# Standard laboratory test groupings with typical cost, time, and physical effort weights
LAB_TESTS = {
    "Atterberg_Limits": {
        "name": "Atterberg Limits (LL, PL, PI)",
        "features": ["LL", "PL", "PI"],
        "cost": 1.0,     # Low cost (relative scale 1-5)
        "time_hr": 2.0,  # ~2 hours testing time
        "effort": 1.0    # Low manual effort
    },
    "Sieve_Analysis": {
        "name": "Grain Size / Sieve Analysis (Fines, Sand, Gravel)",
        "features": ["fines_pct", "sand_pct", "gravel_pct"],
        "cost": 1.5,     # Moderate cost
        "time_hr": 4.0,  # ~4 hours testing time
        "effort": 2.0    # Moderate manual effort
    },
    "Proctor_Compaction": {
        "name": "Standard Proctor Compaction (OMC, MDD)",
        "features": ["OMC", "MDD"],
        "cost": 3.0,     # Higher cost
        "time_hr": 24.0, # ~24 hours with oven drying
        "effort": 4.0    # High physical effort (sample pulverization, compaction blows)
    }
}

class SoilPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names = NUMERIC_FEATURES
        self.train_mean = None
        self.train_cov_inv = None
        
    def fit(self, X_train: pd.DataFrame):
        X_imputed = self.imputer.fit_transform(X_train[self.feature_names])
        self.scaler.fit(X_imputed)
        
        # Precompute domain centroid and inverted covariance for Mahalanobis / domain distance
        X_scaled = self.scaler.transform(X_imputed)
        self.train_mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled, rowvar=False) + 1e-4 * np.eye(X_scaled.shape[1])
        self.train_cov_inv = np.linalg.pinv(cov)
        return self

    def transform(self, X: pd.DataFrame, scale: bool = True):
        # Auto-compute PI if LL and PL are present but PI is missing
        X_df = X.copy()
        if "PI" in X_df.columns and "LL" in X_df.columns and "PL" in X_df.columns:
            mask = X_df["PI"].isna() & (~X_df["LL"].isna()) & (~X_df["PL"].isna())
            X_df.loc[mask, "PI"] = X_df.loc[mask, "LL"] - X_df.loc[mask, "PL"]

        X_imputed = self.imputer.transform(X_df[self.feature_names])
        if scale:
            return self.scaler.transform(X_imputed)
        return X_imputed

    def compute_domain_distance(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculates Mahalanobis distance to training domain centroid.
        Identifies whether a soil sample is out-of-distribution (OOD).
        """
        diff = X_scaled - self.train_mean
        dist = np.sqrt(np.sum((diff @ self.train_cov_inv) * diff, axis=1))
        return dist
