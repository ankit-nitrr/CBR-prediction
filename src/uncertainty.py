"""
Phase IX: Uncertainty & Confidence Quantification Engine for Soil CBR.
Computes:
1. Ensemble Disagreement (std dev between diverse models)
2. Calibrated Conformal Prediction Intervals (held-out residual distribution)
3. Distance-to-Training-Domain (OOD detection)
4. Soil Reliability Index (SRI, 0 - 100%) and Confidence Status (High / Moderate / Low)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.preprocessing import SoilPreprocessor
from src.models import CBRModelEnsemble

class CBRUncertaintyEngine:
    def __init__(self, ensemble: CBRModelEnsemble):
        self.ensemble = ensemble
        self.calibrated_residual_quantile = 2.5 # default fallback
        self.calibration_errors = []
        
    def calibrate(self, df_holdout: pd.DataFrame, coverage_level: float = 0.90):
        """
        Calibrates conformal prediction intervals using absolute residual quantiles
        on held-out data. Ensures 90% empirical coverage.
        """
        preds_dict = self.ensemble.predict_all(df_holdout)
        # Using Random Forest or Ensemble Mean as reference primary predictor
        y_true = df_holdout["CBR"].values
        rf_preds = preds_dict["Random Forest (RF)"]
        
        abs_residuals = np.abs(y_true - rf_preds)
        self.calibration_errors = abs_residuals
        # Conformal quantile: (1 + 1/n) * coverage
        q_level = min(0.99, coverage_level * (1.0 + 1.0 / len(abs_residuals)))
        self.calibrated_residual_quantile = float(np.quantile(abs_residuals, q_level))
        return self

    def predict_with_uncertainty(self, df_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates:
        - Primary CBR prediction (RF / Ensemble Mean)
        - 90% Prediction Interval [Lower CBR, Upper CBR]
        - Prediction Interval Width (PIW)
        - Ensemble Disagreement (Std Dev of models)
        - Distance to Training Domain
        - Soil Reliability Index (SRI, 0-100)
        - Decision Status: ACCEPTABLE vs LOW_CONFIDENCE_VERIFICATION_REQUIRED
        """
        preds_dict = self.ensemble.predict_all(df_sample)
        all_preds = np.array([preds_dict[k] for k in preds_dict]) # shape (5, n_samples)
        
        mean_pred = np.mean(all_preds, axis=0)
        rf_pred = preds_dict["Random Forest (RF)"]
        ensemble_std = np.std(all_preds, axis=0) # Disagreement
        
        # Conformal interval adjusted by local ensemble disagreement and domain distance
        X_scaled = self.ensemble.preprocessor.transform(df_sample, scale=True)
        domain_dist = self.ensemble.preprocessor.compute_domain_distance(X_scaled)
        
        # Scaling factor: if domain distance > 3 or disagreement is high, interval widens
        domain_penalty = np.clip(domain_dist / 3.0, 1.0, 2.5)
        local_margin = self.calibrated_residual_quantile * domain_penalty
        
        lower_bound = np.clip(rf_pred - local_margin, 0.5, 60.0)
        upper_bound = np.clip(rf_pred + local_margin, 0.5, 60.0)
        interval_width = upper_bound - lower_bound
        
        # Compute Soil Reliability Index (SRI) on 0 to 100 scale:
        # High SRI = tight interval, low disagreement, low domain distance
        disagreement_score = np.clip(100.0 - (ensemble_std * 18.0), 10.0, 100.0)
        interval_score = np.clip(100.0 - (interval_width * 10.0), 10.0, 100.0)
        domain_score = np.clip(100.0 - (domain_dist * 15.0), 10.0, 100.0)
        
        sri = 0.35 * disagreement_score + 0.40 * interval_score + 0.25 * domain_score
        sri = np.clip(sri, 5.0, 99.0)
        
        status = []
        for s in sri:
            if s >= 75.0:
                status.append("HIGH_CONFIDENCE (Decision Ready)")
            elif s >= 55.0:
                status.append("MODERATE_CONFIDENCE (Acceptable with caution)")
            else:
                status.append("LOW_CONFIDENCE (Verification Required)")
                
        return {
            "predicted_cbr": rf_pred,
            "ensemble_mean_cbr": mean_pred,
            "cbr_lower_90": lower_bound,
            "cbr_upper_90": upper_bound,
            "interval_width": interval_width,
            "ensemble_std": ensemble_std,
            "domain_distance": domain_dist,
            "sri": sri,
            "status": status,
            "all_model_predictions": preds_dict
        }

    def verify_calibration_on_test(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates the key experiment from Phase IX:
        Do cases classified as low-confidence actually show larger errors on independent data?
        """
        res = self.predict_with_uncertainty(df_test)
        y_true = df_test["CBR"].values
        abs_err = np.abs(y_true - res["predicted_cbr"])
        
        eval_df = pd.DataFrame({
            "actual_cbr": y_true,
            "predicted_cbr": res["predicted_cbr"],
            "abs_error": abs_err,
            "sri": res["sri"],
            "status": res["status"],
            "interval_width": res["interval_width"],
            "domain_distance": res["domain_distance"],
            "covered_in_interval": (y_true >= res["cbr_lower_90"]) & (y_true <= res["cbr_upper_90"])
        })
        
        group_summary = eval_df.groupby("status")["abs_error"].agg(["count", "mean", "std", "median"])
        coverage = float(np.mean(eval_df["covered_in_interval"]) * 100.0)
        mean_width = float(np.mean(eval_df["interval_width"]))
        
        return {
            "eval_df": eval_df,
            "group_summary": group_summary,
            "empirical_coverage_pct": round(coverage, 2),
            "mean_prediction_interval_width": round(mean_width, 2)
        }
