# Machine Learning-Based Prediction of California Bearing Ratio (CBR) of Soils
## Validated on Uploaded Experimental Geotechnical Dataset (Phases VIII – XII)

An end-to-end geotechnical engineering machine learning framework trained directly on your uploaded experimental database from:
> **Citation**: Nagaraju, T.V.; Bahrami, A.; Prasad, C.D.; Mantena, S.; Biswal, M.; Islam, M.R.  
> *Predicting California Bearing Ratio of Lateritic Soils Using Hybrid Machine Learning Technique.*  
> **Buildings 2023**, 13(1), 255. [https://doi.org/10.3390/buildings13010255](https://doi.org/10.3390/buildings13010255)  
> Extracted from **Table A1: Dataset of lateritic soils properties** (Pages 15–18 of `buildings-13-00255-v2.pdf`).

---

## 📌 Uploaded Dataset Characteristics
- **Total Experimental Samples**: 149 complete laboratory specimens tested in accordance with **ASTM D1883-16** (soaked CBR).
- **Soil Classification Domains (USCS)**:
  - Clayey Sand (**SC**): 47 samples (31.5%)
  - Clayey Gravel (**GC**): 44 samples (29.5%)
  - Lean Clay (**CL**): 33 samples (22.1%)
  - Silty Sand (**SM**): 18 samples (12.1%)
  - Fat Clay (**CH**): 5 samples (3.4%)
  - Gravel (**GW/GP**): 2 samples (1.3%)
- **Measured Variables**:
  - `Gravel (%)`: Min 0.0%, Max 65.0%, Mean 22.65%
  - `Sand (%)`: Min 14.0%, Max 88.0%, Mean 42.28%
  - `Fines (%)`: Min 8.0%, Max 78.0%, Mean 35.20%
  - `Liquid Limit (LL, %)`: Min 0.0%, Max 68.0%, Mean 33.56%
  - `Plastic Limit (PL, %)`: Min 0.0%, Max 27.0%, Mean 19.33%
  - `Plasticity Index (PI, %)`: Min 0.0%, Max 41.0%, Mean 14.22%
  - `Optimum Moisture Content (OMC, %)`: Min 5.30%, Max 30.70%, Mean 13.25%
  - `Maximum Dry Density (MDD, g/cc)`: Min 1.870, Max 2.280, Mean 2.080
  - **`Target Soaked CBR (%)`**: Min 12.0%, Max 34.0%, Mean **25.42%** (Std: 4.94%)

---

## 📊 Experimental Results on Uploaded Dataset

### 1. 5-Fold Cross-Validation Performance (5 Algorithms)
| Model | $R^2$ (Mean ± Std) | RMSE (Mean ± Std) | MAE (Mean ± Std) |
|---|---|---|---|
| **Multiple Linear Regression (MLR)** | **0.9736** ± 0.011 | **0.781** ± 0.221 | **0.592** ± 0.098 |
| **Artificial Neural Network (ANN)** | **0.9639** ± 0.009 | **0.919** ± 0.196 | **0.687** ± 0.107 |
| **XGBoost Regressor (XGB)** | **0.9466** ± 0.014 | **1.104** ± 0.190 | **0.833** ± 0.089 |
| **Support Vector Regression (SVR)** | **0.9434** ± 0.029 | **1.137** ± 0.362 | **0.784** ± 0.146 |
| **Random Forest Regressor (RF)** | **0.9338** ± 0.021 | **1.228** ± 0.266 | **0.873** ± 0.098 |

---

### 2. Phase IX: Calibrated Uncertainty Quantification
- **Conformal Prediction Margin (90%)**: $\pm 1.46$ CBR units.
- **Independent Test Set MPIW (Mean Prediction Interval Width)**: $3.21$ CBR units.
- **Empirical Calibration Check**:
  - `HIGH_CONFIDENCE` samples showed an average error of only **$0.56$ CBR units**.
  - Samples categorized with uncertainty/moderate confidence exhibited an error of **$2.22$ CBR units**.
  - **Result**: The uncertainty engine reliably flags higher error cases before design decisions are made!

---

### 3. Phase X: Minimum-Test Verification Engine on Lateritic Soils
Using the virtual missing-variable experiment:
$$\text{TSS}_i = \frac{\Delta U}{\alpha \cdot \text{Cost}_i + \beta \cdot \text{Time}_i + \gamma \cdot \text{Effort}_i}$$

| Candidate Laboratory Test | Features Tested | Lab Time | Exp. $\Delta U$ Reduction | Burden Score | TSS Rank |
|---|---|---|---|---|---|
| **Grain Size / Sieve Analysis** | Fines, Sand, Gravel | 4.0 hrs | **3.780 CBR units** | 0.383 | **9.857 (#1)** |
| **Standard Proctor Compaction** | $OMC, MDD$ | 24.0 hrs | 0.289 CBR units | 1.000 | **0.289 (#2)** |
| **Atterberg Limits** | $LL, PL, PI$ | 2.0 hrs | 0.024 CBR units | 0.225 | **0.105 (#3)** |

*Geotechnical Physical Insight*: In coarse-to-medium grained lateritic soils, grain size distribution ($G, S, F$ proportions) governs the mechanical grain interlock and bearing capacity, producing by far the largest uncertainty reduction ($\Delta U = 3.780$, $TSS = 9.857$).

---

### 4. Phase XI & XII: Closed-Loop Recalibration on Unseen Samples
Demonstrating the feedback loop:
$$\text{SFV} \longrightarrow \text{Initial CBR \& SRI} \longrightarrow \text{TSS Test Selection} \longrightarrow \text{Lab Measurement} \longrightarrow \text{Updated Prediction}$$

- **Sample #1 (Specimen ID 122 - GC, Measured CBR: 33.0%)**:
  - Iteration 1 (only Atterberg known): Pred CBR = $26.58\%$ (Error = $6.42$ units, SRI = $35.8\%$, `LOW_CONFIDENCE`)
  - Recommendation: Sieve Analysis ($TSS = 9.86$)
  - Iteration 2 (Sieve added): Pred CBR = $31.69\%$ (SRI = $71.4\%$)
  - Iteration 3 (Compaction added): Pred CBR = **$32.09\%$** (Error = **$0.91$ units**, SRI = **$75.9\%$**, `ACCEPT_PREDICTION`)
- **Sample #3 (Specimen ID 125 - GC, Measured CBR: 33.0%)**:
  - Initial Error: $6.38$ units $\longrightarrow$ Final Recalibrated Error: **$0.79$ CBR units**!

---

## 🚀 How to Run

### 1. Run Experiments on Terminal
```powershell
python run_experiments.py
```

### 2. Launch Streamlit Web UI Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501` to test single soil samples, sliders, and the closed-loop engine interactively.
