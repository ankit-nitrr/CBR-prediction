"""
Phase X: Minimum-Test Verification Engine.
The primary research innovation:
Determines which laboratory soil test to perform next to achieve maximum
reduction in CBR prediction uncertainty with minimal laboratory cost, time, and physical effort.

Computes:
TSS_i = ΔU / (alpha * cost + beta * time + gamma * effort)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.preprocessing import LAB_TESTS, NUMERIC_FEATURES
from src.uncertainty import CBRUncertaintyEngine

class MinimumTestVerificationEngine:
    def __init__(
        self,
        uncertainty_engine: CBRUncertaintyEngine,
        alpha: float = 0.40,  # Weight for monetary cost
        beta: float = 0.35,   # Weight for test duration (time)
        gamma: float = 0.25   # Weight for manual physical effort
    ):
        self.uncertainty_engine = uncertainty_engine
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.empirical_delta_u_table = {}

    def run_virtual_missing_variable_experiment(
        self, df_validation: pd.DataFrame, n_samples: int = 100
    ) -> pd.DataFrame:
        """
        Executes Steps 16-24 of Phase X Specification:
        1. Take validation observations where all features are known.
        2. Temporarily hide candidate test variables (e.g. OMC/MDD, Fines/Sand, LL/PL).
        3. Predict CBR and calculate initial uncertainty (interval width).
        4. Restore true value and re-predict, calculating reduction in uncertainty (ΔU).
        5. Repeat over validation samples and calculate average ΔU distribution.
        """
        val_subset = df_validation.dropna(subset=NUMERIC_FEATURES).head(n_samples).copy()
        
        # Baseline uncertainty with full features
        baseline_res = self.uncertainty_engine.predict_with_uncertainty(val_subset)
        baseline_widths = baseline_res["interval_width"]
        
        results = []
        for test_key, test_info in LAB_TESTS.items():
            features_to_hide = test_info["features"]
            
            # Mask the candidate test features
            df_masked = val_subset.copy()
            for col in features_to_hide:
                df_masked[col] = np.nan
                
            masked_res = self.uncertainty_engine.predict_with_uncertainty(df_masked)
            masked_widths = masked_res["interval_width"]
            
            # Reduction in uncertainty when test is restored
            delta_u = masked_widths - baseline_widths
            avg_delta_u = float(np.mean(np.maximum(delta_u, 0.0)))
            std_delta_u = float(np.std(delta_u))
            
            # Composite test effort cost denominator (normalized)
            # Cost scale: 1-5, Time scale: 1-24h (normalized by 24h), Effort scale: 1-5
            norm_cost = test_info["cost"] / 3.0
            norm_time = test_info["time_hr"] / 24.0
            norm_effort = test_info["effort"] / 4.0
            
            combined_burden = (
                self.alpha * norm_cost + 
                self.beta * norm_time + 
                self.gamma * norm_effort
            )
            
            tss = avg_delta_u / (combined_burden + 1e-4)
            
            self.empirical_delta_u_table[test_key] = avg_delta_u
            
            results.append({
                "Test_Key": test_key,
                "Test_Name": test_info["name"],
                "Missing_Features": ", ".join(features_to_hide),
                "Cost_Rating": test_info["cost"],
                "Duration_Hours": test_info["time_hr"],
                "Effort_Rating": test_info["effort"],
                "Combined_Burden": round(float(combined_burden), 3),
                "Expected_Delta_U_Reduction": round(avg_delta_u, 3),
                "Delta_U_Std": round(std_delta_u, 3),
                "Test_Selection_Score_TSS": round(float(tss), 3)
            })
            
        summary_df = pd.DataFrame(results).sort_values("Test_Selection_Score_TSS", ascending=False)
        return summary_df

    def recommend_next_test(
        self,
        current_soil_row: pd.Series,
        available_budget_max_tests: int = 2
    ) -> Dict[str, Any]:
        """
        Target-specific test selection for a real soil sample.
        Identifies which candidate test is missing, estimates specific ΔU,
        computes TSS_i, and recommends the optimal test to execute in the lab.
        """
        # Detect missing tests
        candidate_scores = []
        
        # Current uncertainty with currently available features
        sample_df = pd.DataFrame([current_soil_row])
        curr_res = self.uncertainty_engine.predict_with_uncertainty(sample_df)
        curr_width = float(curr_res["interval_width"][0])
        curr_sri = float(curr_res["sri"][0])
        curr_status = curr_res["status"][0]
        
        if curr_sri >= 75.0:
            return {
                "decision": "ACCEPT_PREDICTION",
                "message": f"Prediction reliability is already HIGH (SRI: {curr_sri:.1f}%). No additional laboratory tests required.",
                "candidate_rankings": [],
                "recommended_test": None,
                "current_sri": curr_sri,
                "current_cbr": float(curr_res["predicted_cbr"][0]),
                "current_interval": [float(curr_res["cbr_lower_90"][0]), float(curr_res["cbr_upper_90"][0])]
            }
            
        for test_key, test_info in LAB_TESTS.items():
            feats = test_info["features"]
            # Check if any feature in this test is missing
            is_missing = any(pd.isna(current_soil_row.get(f, np.nan)) for f in feats)
            
            if is_missing:
                expected_delta_u = self.empirical_delta_u_table.get(test_key, 1.2)
                
                norm_cost = test_info["cost"] / 3.0
                norm_time = test_info["time_hr"] / 24.0
                norm_effort = test_info["effort"] / 4.0
                
                combined_burden = (
                    self.alpha * norm_cost + 
                    self.beta * norm_time + 
                    self.gamma * norm_effort
                )
                
                tss = expected_delta_u / (combined_burden + 1e-4)
                
                candidate_scores.append({
                    "test_key": test_key,
                    "test_name": test_info["name"],
                    "features_provided": feats,
                    "expected_delta_u": round(expected_delta_u, 3),
                    "cost_rating": test_info["cost"],
                    "time_hr": test_info["time_hr"],
                    "effort_rating": test_info["effort"],
                    "burden": round(float(combined_burden), 3),
                    "tss": round(float(tss), 3)
                })
                
        if not candidate_scores:
            return {
                "decision": "FULL_SFV_REACHED",
                "message": "All standard soil features are already measured. If CBR confidence remains moderate, proceed to direct laboratory CBR verification.",
                "candidate_rankings": [],
                "recommended_test": None,
                "current_sri": curr_sri,
                "current_cbr": float(curr_res["predicted_cbr"][0]),
                "current_interval": [float(curr_res["cbr_lower_90"][0]), float(curr_res["cbr_upper_90"][0])]
            }
            
        candidate_scores.sort(key=lambda x: x["tss"], reverse=True)
        best_candidate = candidate_scores[0]
        
        return {
            "decision": "CONDUCT_ADDITIONAL_TEST",
            "message": f"Conduct {best_candidate['test_name']} to maximize uncertainty reduction per laboratory effort.",
            "recommended_test": best_candidate,
            "candidate_rankings": candidate_scores,
            "current_sri": curr_sri,
            "current_cbr": float(curr_res["predicted_cbr"][0]),
            "current_interval": [float(curr_res["cbr_lower_90"][0]), float(curr_res["cbr_upper_90"][0])]
        }
