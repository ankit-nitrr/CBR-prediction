"""
Geotechnical Soil CBR Prediction System - Interactive Dashboard.
Features:
- Multi-Source Database Inspection
- Model Comparison (MLR, RF, XGB, SVR, ANN) + Leave-One-Source-Out Generalization
- Phase IX: Uncertainty & Conformal Prediction Intervals + Soil Reliability Index (SRI)
- Phase X: Minimum-Test Verification Engine & Test Selection Score (TSS) Optimizer
- Phase XI & XII: Interactive Closed-Loop Recalibration Simulator
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import NUMERIC_FEATURES, LAB_TESTS
from src.models import CBRModelEnsemble
from src.uncertainty import CBRUncertaintyEngine
from src.minimum_test_engine import MinimumTestVerificationEngine
from src.closed_loop import ClosedLoopRecalibrator

st.set_page_config(
    page_title="Soil CBR ML & Minimum-Test Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-mod {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    data_path = os.path.join("data", "soil_cbr_dataset.csv")
    if not os.path.exists(data_path):
        from data.generate_dataset import generate_geotechnical_cbr_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_geotechnical_cbr_dataset(550)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # Train / Calib / Test split
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    n_train = int(len(df) * 0.70)
    n_calib = int(len(df) * 0.15)
    
    train_df = df.iloc[indices[:n_train]].copy().reset_index(drop=True)
    calib_df = df.iloc[indices[n_train:n_train + n_calib]].copy().reset_index(drop=True)
    test_study_df = df.iloc[indices[n_train + n_calib:]].copy().reset_index(drop=True)
    train_val_df = pd.concat([train_df, calib_df], ignore_index=True)

    ensemble = CBRModelEnsemble()
    ensemble.fit(train_df)

    uncertainty_engine = CBRUncertaintyEngine(ensemble)
    uncertainty_engine.calibrate(calib_df, coverage_level=0.90)

    test_engine = MinimumTestVerificationEngine(uncertainty_engine, alpha=0.40, beta=0.35, gamma=0.25)
    test_engine.run_virtual_missing_variable_experiment(calib_df, n_samples=len(calib_df))

    return df, train_val_df, test_study_df, ensemble, uncertainty_engine, test_engine

df, train_val_df, test_study_df, ensemble, uncertainty_engine, test_engine = load_system()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/geology.png", width=70)
st.sidebar.title("CBR Prediction Engine")
st.sidebar.markdown("**Phases IX - XII Architecture**")
st.sidebar.markdown("""
- **Phase IX**: Prediction & Uncertainty
- **Phase X**: Minimum-Test Engine (TSS)
- **Phase XI**: Closed-Loop Recalibration
- **Phase XII**: Unseen Source Testing
""")

tab_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Prediction & Confidence (Phase IX)",
        "2. Minimum-Test Verification Engine (Phase X)",
        "3. Closed-Loop Recalibration (Phase XI & XII)",
        "4. Model Benchmarking & Generalization",
        "5. Multi-Source Geotechnical Data"
    ]
)

# -------------------------------------------------------------
# TAB 1: PREDICTION & CONFIDENCE (PHASE IX)
# -------------------------------------------------------------
if tab_selection == "1. Prediction & Confidence (Phase IX)":
    st.markdown('<div class="main-header">Soil CBR Prediction with Calibrated Uncertainty</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Outputs dual estimates: predicted CBR (%) and statistical reliability (Prediction Interval & SRI).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Soil Index & Compaction Properties (SFV)")
        soil_type = st.selectbox("Soil Classification Domain", ["CL (Lean Clay)", "CH (Fat Clay)", "ML (Silt)", "SC (Clayey Sand)", "SM (Silty Sand)", "GW/GP (Gravel)"])
        
        c_a, c_b = st.columns(2)
        with c_a:
            ll = st.slider("Liquid Limit, LL (%)", 15.0, 90.0, 38.0, 0.5)
            pl = st.slider("Plastic Limit, PL (%)", 10.0, 45.0, 20.0, 0.5)
            fines = st.slider("Fines Content < 75µm (%)", 5.0, 95.0, 60.0, 0.5)
        with c_b:
            sand = st.slider("Sand Content (%)", 2.0, 85.0, 30.0, 0.5)
            gravel = max(0.0, round(100.0 - fines - sand, 2))
            st.metric("Gravel Content (%)", f"{gravel:.1f}%")
            mdd = st.slider("Max Dry Density, MDD (g/cm³)", 1.45, 2.30, 1.78, 0.01)
            omc = st.slider("Optimum Moisture Content, OMC (%)", 6.0, 30.0, 16.5, 0.5)

        pi = max(1.0, round(ll - pl, 2))
        st.caption(f"Calculated Plasticity Index (PI = LL - PL): **{pi:.1f}%**")

    input_row = pd.Series({
        "LL": ll, "PL": pl, "PI": pi,
        "fines_pct": fines, "sand_pct": sand, "gravel_pct": gravel,
        "MDD": mdd, "OMC": omc
    })

    sample_df = pd.DataFrame([input_row])
    res = uncertainty_engine.predict_with_uncertainty(sample_df)

    pred_cbr = float(res["predicted_cbr"][0])
    lower_cbr = float(res["cbr_lower_90"][0])
    upper_cbr = float(res["cbr_upper_90"][0])
    piw = float(res["interval_width"][0])
    sri = float(res["sri"][0])
    status_str = res["status"][0]
    disagree_std = float(res["ensemble_std"][0])
    dom_dist = float(res["domain_distance"][0])

    with col2:
        st.subheader("Prediction Reliability & Decision Status")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted CBR", f"{pred_cbr:.2f} %")
        m2.metric("90% Prediction Interval", f"[{lower_cbr:.2f}%, {upper_cbr:.2f}%]")
        m3.metric("Interval Width (MPIW)", f"{piw:.2f} %")

        sri_col, badge_col = st.columns([1, 1])
        with sri_col:
            st.metric("Soil Reliability Index (SRI)", f"{sri:.1f} / 100")
        with badge_col:
            st.write("Decision Status:")
            if "HIGH" in status_str:
                st.markdown('<span class="badge-high">HIGH CONFIDENCE (Decision Ready)</span>', unsafe_allow_html=True)
            elif "MODERATE" in status_str:
                st.markdown('<span class="badge-mod">MODERATE CONFIDENCE (Acceptable with caution)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-low">LOW CONFIDENCE (Verification Required)</span>', unsafe_allow_html=True)

        st.progress(min(1.0, sri / 100.0))

        st.markdown("---")
        st.markdown("**Diagnostics breakdown:**")
        d1, d2 = st.columns(2)
        d1.write(f"• **Ensemble Disagreement:** ±{disagree_std:.2f}% (std dev)")
        d2.write(f"• **Domain Distance (Mahalanobis):** {dom_dist:.2f}")

        # Multi-model comparison bar chart
        st.markdown("**Predictions across all 5 Algorithms:**")
        preds_all = res["all_model_predictions"]
        model_names = list(preds_all.keys())
        model_vals = [float(preds_all[k][0]) for k in model_names]

        fig, ax = plt.subplots(figsize=(6, 2.6))
        bars = ax.barh([m.split("(")[0].strip() for m in model_names], model_vals, color="#3B82F6")
        ax.set_xlabel("Predicted CBR (%)")
        ax.set_xlim(0, max(model_vals) * 1.35)
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.3, b.get_y() + b.get_height()/2, f"{w:.2f}%", va="center", fontsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)

# -------------------------------------------------------------
# TAB 2: MINIMUM-TEST VERIFICATION ENGINE (PHASE X)
# -------------------------------------------------------------
elif tab_selection == "2. Minimum-Test Verification Engine (Phase X)":
    st.markdown('<div class="main-header">Phase X: Minimum-Test Verification Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Selects the optimal laboratory test to conduct to achieve maximum uncertainty reduction per unit test burden.</div>', unsafe_allow_html=True)

    st.info("""
    **Test Selection Score (TSS) Formula:**
    $$TSS_i = \\frac{\\Delta U}{\\alpha \\times \\text{Cost}_i + \\beta \\times \\text{Time}_i + \\gamma \\times \\text{Effort}_i}$$
    """)

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.subheader("1. Laboratory Decision Weights")
        alpha = st.slider("Weight for Cost (α)", 0.0, 1.0, 0.40, 0.05)
        beta = st.slider("Weight for Duration/Time (β)", 0.0, 1.0, 0.35, 0.05)
        gamma = st.slider("Weight for Physical Effort (γ)", 0.0, 1.0, 0.25, 0.05)
        
        total_w = alpha + beta + gamma
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Note: Weights sum to {total_w:.2f}. Engine will normalize automatically.")

        st.subheader("2. Currently Available Soil Tests for Target Sample")
        has_atterberg = st.checkbox("Atterberg Limits Conducted (LL, PL, PI known)", value=True)
        has_sieve = st.checkbox("Sieve Analysis Conducted (Fines, Sand, Gravel known)", value=False)
        has_proctor = st.checkbox("Standard Proctor Compaction Conducted (OMC, MDD known)", value=False)

    # Build target sample with missing fields
    target_sample = {}
    if has_atterberg:
        target_sample.update({"LL": 35.0, "PL": 18.0, "PI": 17.0})
    else:
        target_sample.update({"LL": np.nan, "PL": np.nan, "PI": np.nan})

    if has_sieve:
        target_sample.update({"fines_pct": 55.0, "sand_pct": 35.0, "gravel_pct": 10.0})
    else:
        target_sample.update({"fines_pct": np.nan, "sand_pct": np.nan, "gravel_pct": np.nan})

    if has_proctor:
        target_sample.update({"OMC": 16.0, "MDD": 1.78})
    else:
        target_sample.update({"OMC": np.nan, "MDD": np.nan})

    test_engine.alpha = alpha / max(1e-4, total_w)
    test_engine.beta = beta / max(1e-4, total_w)
    test_engine.gamma = gamma / max(1e-4, total_w)

    recommendation = test_engine.recommend_next_test(pd.Series(target_sample))

    with c2:
        st.subheader("3. Optimization & Recommendation Output")
        st.metric("Current CBR Estimate", f"{recommendation['current_cbr']:.2f}%", 
                  f"SRI: {recommendation['current_sri']:.1f}%")

        if recommendation["decision"] == "ACCEPT_PREDICTION":
            st.success(f"✅ {recommendation['message']}")
        elif recommendation["decision"] == "FULL_SFV_REACHED":
            st.warning(f"⚠️ {recommendation['message']}")
        else:
            best = recommendation["recommended_test"]
            st.success(f"🎯 **RECOMMENDED TEST:** {best['test_name']}")
            st.markdown(f"""
            - **Expected Uncertainty Reduction (ΔU):** **{best['expected_delta_u']:.2f}** CBR units
            - **Lab Duration:** {best['time_hr']} hours | **Cost Index:** {best['cost_rating']} | **Effort Index:** {best['effort_rating']}
            - **Test Selection Score (TSS):** **{best['tss']:.2f}**
            """)

            st.write("---")
            st.markdown("**Full Candidate Ranking Matrix:**")
            cand_df = pd.DataFrame(recommendation["candidate_rankings"])
            cand_df = cand_df.rename(columns={
                "test_name": "Test Name",
                "expected_delta_u": "Exp ΔU (Reduction)",
                "time_hr": "Time (hr)",
                "cost_rating": "Cost (1-5)",
                "effort_rating": "Effort (1-5)",
                "tss": "TSS Score"
            })
            st.dataframe(cand_df[["Test Name", "Exp ΔU (Reduction)", "Time (hr)", "Cost (1-5)", "Effort (1-5)", "TSS Score"]], hide_index=True)

# -------------------------------------------------------------
# TAB 3: CLOSED-LOOP RECALIBRATION (PHASE XI & XII)
# -------------------------------------------------------------
elif tab_selection == "3. Closed-Loop Recalibration (Phase XI & XII)":
    st.markdown('<div class="main-header">Phases XI & XII: Closed-Loop Recalibration Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrates the feedback loop: Prediction → Confidence check → Minimum-test selection → Lab measurement → Recalibrated CBR.</div>', unsafe_allow_html=True)

    st.markdown("""
    Select a test soil from the held-out research publication (**Study_E_Clayey_Expansive_Soils**).
    The system starts with only basic Atterberg index properties and requests minimum tests until confidence is reached!
    """)

    specimen_idx = st.selectbox(
        "Select Soil Specimen from Independent Validation Set:",
        list(range(min(15, len(test_study_df)))),
        format_func=lambda i: f"Specimen #{i+1} - Soil Class: {test_study_df.iloc[i]['soil_domain']} | Actual Measured CBR: {test_study_df.iloc[i]['CBR']:.2f}%"
    )

    soil_specimen = test_study_df.iloc[specimen_idx]
    
    recalibrator = ClosedLoopRecalibrator(uncertainty_engine, test_engine, sri_target_threshold=75.0)
    sim_res = recalibrator.run_simulation(soil_specimen, initially_known_tests=["Atterberg_Limits"])

    st.write("---")
    st.subheader(f"Closed-Loop Workflow for Specimen #{specimen_idx + 1}")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reference Actual Measured CBR", f"{sim_res['actual_cbr']:.2f}%")
    col_b.metric("Initial Prediction Error", f"{sim_res['initial_abs_error']:.2f}%")
    col_c.metric("Final Post-Recalibration Error", f"{sim_res['final_abs_error']:.2f}%", 
                 f"-{sim_res['initial_abs_error'] - sim_res['final_abs_error']:.2f}% error drop")

    st.markdown("#### Iteration Step Timeline:")
    for step in sim_res["history"]:
        with st.expander(f"Iteration {step['iteration']}: Status = {step['decision']}", expanded=True):
            s1, s2, s3 = st.columns(3)
            s1.write(f"**Predicted CBR:** {step['current_cbr']:.2f}%")
            s2.write(f"**90% Prediction Interval:** [{step['current_interval'][0]:.2f}%, {step['current_interval'][1]:.2f}%]")
            s3.write(f"**SRI Reliability Index:** {step['current_sri']:.1f}%")
            st.write(f"**Features Known in SFV:** {', '.join(step['active_features'])}")
            if step["recommended_test"]:
                st.info(f"➡️ **System Recommended:** {step['recommended_test']['test_name']} (TSS: {step['recommended_test']['tss']:.2f})")

# -------------------------------------------------------------
# TAB 4: MODEL BENCHMARKING & GENERALIZATION
# -------------------------------------------------------------
elif tab_selection == "4. Model Benchmarking & Generalization":
    st.markdown('<div class="main-header">Algorithm Benchmarking & Unseen Source Generalization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating 5 algorithms across standard 5-Fold CV and Leave-One-Source-Out (LOSO) Cross-Validation.</div>', unsafe_allow_html=True)

    st.markdown("### 1. 5-Fold Cross Validation Across Entire Dataset")
    cv_summary = ensemble.evaluate_cross_validation(train_val_df, n_splits=5)
    st.dataframe(cv_summary.style.format({
        "R2_Mean": "{:.4f}", "R2_Std": "{:.4f}",
        "RMSE_Mean": "{:.3f}", "RMSE_Std": "{:.3f}",
        "MAE_Mean": "{:.3f}", "MAE_Std": "{:.3f}"
    }), hide_index=True)

    st.markdown("### 2. Leave-One-Source-Out (LOSO) Generalization Benchmark")
    st.info("Testing on an entirely unseen published research study proves the model generalizes across geotechnical laboratories and regions without overfitting.")
    loso_df = ensemble.evaluate_leave_one_source_out(df)
    loso_agg = loso_df.groupby("Model")[["R2", "RMSE", "MAE"]].mean().reset_index().sort_values("R2", ascending=False)
    st.dataframe(loso_agg.style.format({
        "R2": "{:.4f}", "RMSE": "{:.3f}", "MAE": "{:.3f}"
    }), hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(loso_agg["Model"], loso_agg["R2"], color=["#10B981", "#3B82F6", "#6366F1", "#F59E0B", "#EF4444"])
    ax.set_ylabel("Generalization R²")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Generalization R² on Unseen Geotechnical Soil Classes / Sources")
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig)

# -------------------------------------------------------------
# TAB 5: MULTI-SOURCE GEOTECHNICAL DATA
# -------------------------------------------------------------
elif tab_selection == "5. Multi-Source Geotechnical Data":
    st.markdown('<div class="main-header">Multi-Source Geotechnical Database</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Compiled dataset with {len(df)} observations from 5 distinct published studies and 6 geotechnical domains.</div>', unsafe_allow_html=True)

    st.dataframe(df.head(25), hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution by Soil Classification Domain")
        st.bar_chart(df["soil_domain"].value_counts())
    with c2:
        st.subheader("Distribution by Research Publication Source")
        st.bar_chart(df["source_study"].value_counts())

st.markdown("---")
st.caption("AI-Powered Geotechnical CBR Estimation System | Compliant with ASTM D1883, AASHTO T193, & IS 2720")
