"""
Dataset generator for Soil California Bearing Ratio (CBR) prediction.
Synthesizes a realistic, multi-source geotechnical dataset grounded in empirical
geotechnical relationships (ASTM D1883, AASHTO T193, IS 2720) and published
literature (e.g. Nagaraj & Suresh, Taskiran, Onyelowe et al., Tenpe & Patel).
"""

import numpy as np
import pandas as pd
import os

def generate_geotechnical_cbr_dataset(n_samples=550, random_state=42):
    np.random.seed(random_state)
    
    sources = [
        {"name": "Study_A_Highway_Subgrades", "region": "South Asia", "bias": 0.0, "n": 130},
        {"name": "Study_B_AASHTO_Pavements", "region": "North America", "bias": 0.4, "n": 120},
        {"name": "Study_C_Tropical_Laterites", "region": "Sub-Saharan Africa", "bias": -0.3, "n": 100},
        {"name": "Study_D_Arid_Granular_Subgrades", "region": "Middle East", "bias": 0.6, "n": 100},
        {"name": "Study_E_Clayey_Expansive_Soils", "region": "East Asia", "bias": -0.5, "n": 100},
    ]
    
    records = []
    
    for src in sources:
        n_src = src["n"]
        for _ in range(n_src):
            # Soil classification domain
            soil_class = np.random.choice(
                ["CL", "CH", "ML", "SC", "SM", "GW/GP"], 
                p=[0.30, 0.20, 0.15, 0.15, 0.12, 0.08]
            )
            
            if soil_class == "CH": # Fat / High plasticity Clay
                ll = np.random.uniform(50.0, 85.0)
                pl = np.random.uniform(22.0, 36.0)
                fines = np.random.uniform(65.0, 96.0)
                sand = np.random.uniform(4.0, 28.0)
                gravel = max(0.0, 100.0 - fines - sand)
                mdd = np.random.uniform(1.50, 1.75) # g/cm3
                omc = np.random.uniform(18.0, 28.0) # %
                base_cbr = np.random.uniform(1.8, 4.5)
            elif soil_class == "CL": # Lean Clay
                ll = np.random.uniform(28.0, 49.0)
                pl = np.random.uniform(14.0, 24.0)
                fines = np.random.uniform(50.0, 80.0)
                sand = np.random.uniform(15.0, 40.0)
                gravel = max(0.0, 100.0 - fines - sand)
                mdd = np.random.uniform(1.68, 1.90)
                omc = np.random.uniform(13.0, 20.0)
                base_cbr = np.random.uniform(3.5, 8.5)
            elif soil_class == "ML": # Silt
                ll = np.random.uniform(24.0, 45.0)
                pl = np.random.uniform(18.0, 30.0)
                fines = np.random.uniform(55.0, 85.0)
                sand = np.random.uniform(12.0, 35.0)
                gravel = max(0.0, 100.0 - fines - sand)
                mdd = np.random.uniform(1.65, 1.85)
                omc = np.random.uniform(14.0, 21.0)
                base_cbr = np.random.uniform(3.0, 7.0)
            elif soil_class == "SC": # Clayey Sand
                ll = np.random.uniform(20.0, 40.0)
                pl = np.random.uniform(12.0, 20.0)
                fines = np.random.uniform(20.0, 45.0)
                sand = np.random.uniform(45.0, 70.0)
                gravel = max(0.0, 100.0 - fines - sand)
                mdd = np.random.uniform(1.82, 2.05)
                omc = np.random.uniform(10.0, 15.0)
                base_cbr = np.random.uniform(7.0, 16.0)
            elif soil_class == "SM": # Silty Sand
                ll = np.random.uniform(18.0, 35.0)
                pl = np.random.uniform(12.0, 22.0)
                fines = np.random.uniform(15.0, 40.0)
                sand = np.random.uniform(50.0, 75.0)
                gravel = max(0.0, 100.0 - fines - sand)
                mdd = np.random.uniform(1.85, 2.10)
                omc = np.random.uniform(9.0, 14.0)
                base_cbr = np.random.uniform(9.0, 20.0)
            else: # Well/Poorly Graded Gravel / Sand (GW/GP)
                ll = np.random.uniform(15.0, 28.0)
                pl = np.random.uniform(10.0, 18.0)
                fines = np.random.uniform(3.0, 15.0)
                sand = np.random.uniform(30.0, 55.0)
                gravel = max(0.0, 100.0 - fines - sand)
                mdd = np.random.uniform(2.00, 2.25)
                omc = np.random.uniform(6.5, 11.0)
                base_cbr = np.random.uniform(18.0, 38.0)

            pi = max(1.0, ll - pl)
            
            # Geotechnical empirical relation with non-linear terms & noise
            # CBR decreases with higher OMC, PI, Fines; increases with MDD, Sand/Gravel
            # Formula reflects standard geotechnical regression tendencies + interaction:
            cbr_calc = (
                base_cbr 
                + 12.0 * (mdd - 1.80) 
                - 0.45 * (omc - 15.0) 
                - 0.15 * (pi - 15.0) 
                - 0.08 * (fines - 50.0) 
                + 0.05 * (sand - 30.0)
                + 0.07 * (gravel - 10.0)
                + src["bias"]
                + np.random.normal(0.0, 0.65) # measurement noise in laboratory CBR
            )
            
            cbr = round(float(np.clip(cbr_calc, 1.2, 45.0)), 2)
            
            records.append({
                "source_study": src["name"],
                "geographical_region": src["region"],
                "test_condition": "Soaked (96-hr)",
                "soil_domain": soil_class,
                "LL": round(float(ll), 2),
                "PL": round(float(pl), 2),
                "PI": round(float(pi), 2),
                "fines_pct": round(float(fines), 2),
                "sand_pct": round(float(sand), 2),
                "gravel_pct": round(float(gravel), 2),
                "MDD": round(float(mdd), 3),
                "OMC": round(float(omc), 2),
                "CBR": cbr
            })
            
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_geotechnical_cbr_dataset(n_samples=550)
    csv_path = os.path.join("data", "soil_cbr_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} multi-source soil CBR records saved to {csv_path}")
    print(df.head())
    print("\nSoil Domain Distribution:")
    print(df["soil_domain"].value_counts())
    print("\nSource Distribution:")
    print(df["source_study"].value_counts())
