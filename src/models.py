"""
Machine Learning Models Suite for Soil California Bearing Ratio (CBR) Prediction.
Implements the 5 core algorithms:
1. Multiple Linear Regression (MLR)
2. Random Forest Regressor (RF)
3. XGBoost Regressor (XGB)
4. Support Vector Regressor (SVR)
5. Artificial Neural Network / Multilayer Perceptron (ANN)

Includes standard 5-fold CV and Leave-One-Source-Out (LOSO) cross-validation
to evaluate model reliability on unseen geotechnical research studies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from src.preprocessing import NUMERIC_FEATURES, SoilPreprocessor

def get_base_models() -> Dict[str, Any]:
    """Returns initialized regression models configured for soil CBR prediction."""
    return {
        "Multiple Linear Regression (MLR)": LinearRegression(),
        "Random Forest (RF)": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_split=4, random_state=42
        ),
        "XGBoost (XGB)": XGBRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.08, subsample=0.85, random_state=42
        ),
        "Support Vector Regression (SVR)": SVR(
            C=10.0, epsilon=0.2, kernel="rbf"
        ),
        "Artificial Neural Network (ANN)": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=800, activation="relu",
            learning_rate_init=0.005, random_state=42
        )
    }

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes R², RMSE, and MAE."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}

class CBRModelEnsemble:
    """
    Manages training, cross-validation, and multi-model predictions for CBR.
    """
    def __init__(self):
        self.preprocessor = SoilPreprocessor()
        self.models = get_base_models()
        self.fitted_models = {}
        self.is_fitted = False
        self.held_out_residuals = {}

    def fit(self, df_train: pd.DataFrame):
        self.preprocessor.fit(df_train)
        X_scaled = self.preprocessor.transform(df_train, scale=True)
        y_train = df_train["CBR"].values
        
        self.fitted_models = {}
        for name, model in self.models.items():
            model.fit(X_scaled, y_train)
            self.fitted_models[name] = model
            
        self.is_fitted = True
        return self

    def predict_all(self, df_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Runs predictions on all 5 algorithms."""
        if not self.is_fitted:
            raise RuntimeError("Models must be fitted before predict_all.")
            
        X_scaled = self.preprocessor.transform(df_test, scale=True)
        predictions = {}
        for name, model in self.fitted_models.items():
            preds = model.predict(X_scaled)
            # CBR cannot be physically negative
            predictions[name] = np.clip(preds, 0.5, 60.0)
        return predictions

    def evaluate_cross_validation(self, df: pd.DataFrame, n_splits=5) -> pd.DataFrame:
        """Standard K-Fold Cross Validation across all models."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {name: {"R2": [], "RMSE": [], "MAE": []} for name in self.models}
        
        X_raw = df[NUMERIC_FEATURES]
        y_raw = df["CBR"].values
        
        for train_idx, val_idx in kf.split(df):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            
            prep = SoilPreprocessor().fit(train_df)
            X_tr = prep.transform(train_df, scale=True)
            y_tr = train_df["CBR"].values
            X_vl = prep.transform(val_df, scale=True)
            y_vl = val_df["CBR"].values
            
            for name, model_cls in get_base_models().items():
                m = model_cls.fit(X_tr, y_tr)
                p = np.clip(m.predict(X_vl), 0.5, 60.0)
                metrics = evaluate_predictions(y_vl, p)
                results[name]["R2"].append(metrics["R2"])
                results[name]["RMSE"].append(metrics["RMSE"])
                results[name]["MAE"].append(metrics["MAE"])
                
        summary = []
        for name, m_dict in results.items():
            summary.append({
                "Model": name,
                "R2_Mean": np.mean(m_dict["R2"]),
                "R2_Std": np.std(m_dict["R2"]),
                "RMSE_Mean": np.mean(m_dict["RMSE"]),
                "RMSE_Std": np.std(m_dict["RMSE"]),
                "MAE_Mean": np.mean(m_dict["MAE"]),
                "MAE_Std": np.std(m_dict["MAE"])
            })
        return pd.DataFrame(summary).sort_values("R2_Mean", ascending=False)

    def evaluate_leave_one_source_out(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Leave-One-Source-Out (LOSO) Cross-Validation:
        Trains on 4 research sources, tests strictly on the 5th unseen research source.
        Directly satisfies Project Specification Phase VIII & IX.
        """
        # If multi-source, split by source_study; otherwise split by soil_domain
        group_col = "source_study" if df["source_study"].nunique() > 1 else "soil_domain"
        # Filter groups with at least 10 samples
        valid_groups = [g for g, c in df[group_col].value_counts().items() if c >= 10]
        loso_records = []
        
        for grp in valid_groups:
            train_df = df[df[group_col] != grp]
            test_df = df[df[group_col] == grp]
            
            prep = SoilPreprocessor().fit(train_df)
            X_tr = prep.transform(train_df, scale=True)
            y_tr = train_df["CBR"].values
            X_ts = prep.transform(test_df, scale=True)
            y_ts = test_df["CBR"].values
            
            for name, model_cls in get_base_models().items():
                m = model_cls.fit(X_tr, y_tr)
                p = np.clip(m.predict(X_ts), 0.5, 60.0)
                metrics = evaluate_predictions(y_ts, p)
                loso_records.append({
                    "Held_Out_Group": grp,
                    "Model": name,
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "N_Test_Samples": len(test_df)
                })
                
        return pd.DataFrame(loso_records)
