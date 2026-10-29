"""
Master Execution Script for Soil California Bearing Ratio (CBR) ML System.
Trained and evaluated directly on the USER'S UPLOADED GEOTECHNICAL DATASET:
'buildings-13-00255-v2.pdf' (Table A1, 149 experimental lateritic soil samples).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from src.preprocessing import NUMERIC_FEATURES
from src.models import CBRModelEnsemble, get_base_models, evaluate_predictions
from src.uncertainty import CBRUncertaintyEngine
from src.minimum_test_engine import MinimumTestVerificationEngine
from src.closed_loop import ClosedLoopRecalibrator

def main():
    print("=" * 80)
    print("GEOTECHNICAL CBR MACHINE LEARNING SYSTEM - UPLOADED DATASET BENCHMARK")
    print("Source: Table A1 of 'buildings-13-00255-v2.pdf' (Nagaraju et al., Buildings 2023)")
    print("=" * 80)

    # 1. Load Uploaded Dataset
    csv_path = os.path.join("data", "soil_cbr_dataset.csv")
    if not os.path.exists(csv_path):
        from data.extract_uploaded_dataset import extract_pdf_dataset
        df = extract_pdf_dataset("buildings-13-00255-v2.pdf")
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    print(f"\n[1] UPLOADED DATASET OVERVIEW:")
    print(f"Total Experimental Samples: {len(df)}")
    print(f"Target CBR Range (Soaked): {df['CBR'].min():.1f}% to {df['CBR'].max():.1f}% (Mean: {df['CBR'].mean():.2f}%, Std: {df['CBR'].std():.2f}%)")
    print("\nSoil Domain Distribution:")
    for dom, count in df['soil_domain'].value_counts().items():
        print(f"  • {dom}: {count} samples ({count/len(df)*100:.1f}%)")

    # 2. Train-Calibration-Test Split (70% Train, 15% Calibration, 15% Independent Test)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    n_train = int(len(df) * 0.70)
    n_calib = int(len(df) * 0.15)
    
    train_idx = indices[:n_train]
    calib_idx = indices[n_train:n_train + n_calib]
    test_idx = indices[n_train + n_calib:]

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    calib_df = df.iloc[calib_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    print(f"\n[2] DATA SPLITTING:")
    print(f"Training Set:       {len(train_df)} samples (70%)")
    print(f"Calibration Set:    {len(calib_df)} samples (15%) - For Phase IX Conformal Prediction Calibration")
    print(f"Independent Test:   {len(test_df)} samples (15%) - For Phase XII Laboratory Generalization Validation")

    # 3. 5-Fold Cross Validation across the 5 ML Algorithms
    print("\n" + "=" * 80)
    print("[3] 5-FOLD CROSS VALIDATION ACROSS 5 ALGORITHMS (R², RMSE, MAE):")
    print("=" * 80)
    ensemble = CBRModelEnsemble()
    cv_summary = ensemble.evaluate_cross_validation(df, n_splits=5)
    print(cv_summary.to_string(index=False))

    # 4. Domain-Wise / Soil Class Generalization Benchmark
    print("\n" + "=" * 80)
    print("[4] SOIL DOMAIN GENERALIZATION BENCHMARK (LEAVE-ONE-DOMAIN-OUT):")
    print("Evaluates reliability when predicting across distinct geotechnical soil classes.")
    print("=" * 80)
    major_domains = [d for d, c in df['soil_domain'].value_counts().items() if c >= 15]
    domain_records = []
    
    for dom in major_domains:
        d_tr = df[df['soil_domain'] != dom]
        d_ts = df[df['soil_domain'] == dom]
        
        from src.preprocessing import SoilPreprocessor
        p = SoilPreprocessor().fit(d_tr)
        X_tr = p.transform(d_tr, scale=True)
        y_tr = d_tr['CBR'].values
        X_ts = p.transform(d_ts, scale=True)
        y_ts = d_ts['CBR'].values
        
        for name, m_cls in get_base_models().items():
            m = m_cls.fit(X_tr, y_tr)
            preds = np.clip(m.predict(X_ts), 0.5, 60.0)
            metrics = evaluate_predictions(y_ts, preds)
            domain_records.append({
                "Held_Out_Domain": dom,
                "Model": name,
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "N_Samples": len(d_ts)
            })
            
    domain_df = pd.DataFrame(domain_records)
    domain_agg = domain_df.groupby("Model")[["R2", "RMSE", "MAE"]].mean().reset_index().sort_values("R2", ascending=False)
    print(domain_agg.to_string(index=False))

    # Fit final ensemble on training set
    ensemble.fit(train_df)

    # 5. Phase IX: Uncertainty Quantification
    print("\n" + "=" * 80)
    print("[5] PHASE IX - UNCERTAINTY / CONFIDENCE MODEL EVALUATION:")
    print("=" * 80)
    uncertainty_engine = CBRUncertaintyEngine(ensemble)
    uncertainty_engine.calibrate(calib_df, coverage_level=0.90)
    print(f"Conformal Prediction Residual Margin (90%): ±{uncertainty_engine.calibrated_residual_quantile:.2f} CBR units")

    # Verify on independent test set
    test_eval = uncertainty_engine.verify_calibration_on_test(test_df)
    print(f"\nIndependent Test Set Empirical Coverage (Target 90%): {test_eval['empirical_coverage_pct']}%")
    print(f"Mean Prediction Interval Width (MPIW): {test_eval['mean_prediction_interval_width']:.2f} CBR units")
    print("\nEmpirical Calibration Verification:")
    print(test_eval["group_summary"].to_string())

    # 6. Phase X: Minimum-Test Verification Engine
    print("\n" + "=" * 80)
    print("[6] PHASE X - MINIMUM-TEST VERIFICATION ENGINE (VIRTUAL MISSING-VARIABLE SIMULATION):")
    print("Calculates Expected Uncertainty Reduction (Delta_U) and Test Selection Score (TSS)")
    print("=" * 80)
    test_engine = MinimumTestVerificationEngine(uncertainty_engine, alpha=0.40, beta=0.35, gamma=0.25)
    tss_matrix = test_engine.run_virtual_missing_variable_experiment(calib_df, n_samples=len(calib_df))
    print(tss_matrix.to_string(index=False))

    # 7. Phase XI & XII: Closed-Loop Recalibration
    print("\n" + "=" * 80)
    print("[7] PHASE XI & XII - CLOSED-LOOP RECALIBRATION ON REAL SOIL SAMPLES:")
    print("=" * 80)
    recalibrator = ClosedLoopRecalibrator(uncertainty_engine, test_engine, sri_target_threshold=75.0)

    # Pick 3 diverse soils from independent test set
    sample_soils = test_df.sample(3, random_state=42)
    s_idx = 1
    for _, soil_row in sample_soils.iterrows():
        print(f"\n--- Sample #{s_idx} (ID {int(soil_row['S_No'])}) | Class: {soil_row['soil_domain']} | Measured CBR: {soil_row['CBR']:.1f}% ---")
        sim_res = recalibrator.run_simulation(soil_row, initially_known_tests=["Atterberg_Limits"])
        
        for step in sim_res["history"]:
            print(f" Iteration {step['iteration']}: Predicted CBR = {step['current_cbr']:.2f}% "
                  f"(90% CI: [{step['current_interval'][0]:.2f}, {step['current_interval'][1]:.2f}], "
                  f"Width = {step['interval_width']:.2f}) | SRI = {step['current_sri']:.1f}% | Status: {step['decision']}")
            if step["recommended_test"]:
                print(f"   --> Recommended Test: {step['recommended_test']['test_name']} (TSS: {step['recommended_test']['tss']:.2f})")
                
        print(f" Final Recalibration Result:")
        print(f"   Initial Error: {sim_res['initial_abs_error']:.2f} CBR units")
        print(f"   Final Error:   {sim_res['final_abs_error']:.2f} CBR units")
        print(f"   Error Reduction: {sim_res['error_reduction']:.2f} CBR units")
        s_idx += 1

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY ON UPLOADED DATASET!")
    print("=" * 80)

if __name__ == "__main__":
    main()
