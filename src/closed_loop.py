"""
Phase XI & XII: Closed-Loop Recalibration & Laboratory Validation Workflow.
Implements the central feedback loop:
Prediction -> Confidence Check -> Minimum Test Selection -> Recalibration -> Final Decision.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.uncertainty import CBRUncertaintyEngine
from src.minimum_test_engine import MinimumTestVerificationEngine

class ClosedLoopRecalibrator:
    def __init__(
        self,
        uncertainty_engine: CBRUncertaintyEngine,
        test_engine: MinimumTestVerificationEngine,
        sri_target_threshold: float = 75.0,
        max_test_iterations: int = 3
    ):
        self.uncertainty_engine = uncertainty_engine
        self.test_engine = test_engine
        self.sri_target_threshold = sri_target_threshold
        self.max_test_iterations = max_test_iterations

    def run_simulation(
        self,
        ground_truth_soil: pd.Series,
        initially_known_tests: List[str] = ["Atterberg_Limits"]
    ) -> Dict[str, Any]:
        """
        Simulates the full closed-loop laboratory recalibration for a soil specimen.
        Demonstrates the shrinkage of prediction intervals and rise in SRI.
        """
        # Build initial partial Soil Feature Vector (SFV)
        sfv = {}
        for k in ["LL", "PL", "PI", "fines_pct", "sand_pct", "gravel_pct", "MDD", "OMC"]:
            sfv[k] = np.nan
            
        from src.preprocessing import LAB_TESTS
        for t_key in initially_known_tests:
            if t_key in LAB_TESTS:
                for f in LAB_TESTS[t_key]["features"]:
                    sfv[f] = ground_truth_soil.get(f, np.nan)
                    
        history = []
        iteration = 0
        current_series = pd.Series(sfv)
        
        while iteration < self.max_test_iterations:
            iteration += 1
            rec = self.test_engine.recommend_next_test(current_series)
            
            step_record = {
                "iteration": iteration,
                "current_cbr": rec["current_cbr"],
                "current_interval": rec["current_interval"],
                "interval_width": rec["current_interval"][1] - rec["current_interval"][0],
                "current_sri": rec["current_sri"],
                "decision": rec["decision"],
                "message": rec["message"],
                "active_features": [k for k, v in current_series.items() if not pd.isna(v)],
                "recommended_test": rec.get("recommended_test", None)
            }
            history.append(step_record)
            
            if rec["decision"] != "CONDUCT_ADDITIONAL_TEST" or rec["recommended_test"] is None:
                break
                
            # Simulate lab technician performing the recommended test:
            # Reveal actual laboratory values for the recommended features
            test_info = rec["recommended_test"]
            for f in test_info["features_provided"]:
                current_series[f] = ground_truth_soil.get(f, np.nan)
                
        actual_cbr = ground_truth_soil.get("CBR", np.nan)
        initial_error = abs(history[0]["current_cbr"] - actual_cbr) if not pd.isna(actual_cbr) else None
        final_error = abs(history[-1]["current_cbr"] - actual_cbr) if not pd.isna(actual_cbr) else None
        
        return {
            "history": history,
            "initial_step": history[0],
            "final_step": history[-1],
            "actual_cbr": actual_cbr,
            "initial_abs_error": initial_error,
            "final_abs_error": final_error,
            "error_reduction": (initial_error - final_error) if initial_error is not None and final_error is not None else None,
            "interval_shrinkage_pct": round(
                (1.0 - (history[-1]["interval_width"] / (history[0]["interval_width"] + 1e-4))) * 100.0, 2
            )
        }
