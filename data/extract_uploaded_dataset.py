"""
Extracts the uploaded real geotechnical dataset from 'buildings-13-00255-v2.pdf'
(Table A1: Dataset of lateritic soils properties, Pages 15-18).
Citation: Nagaraju, T.V.; Bahrami, A.; Prasad, C.D.; et al.
Predicting California Bearing Ratio of Lateritic Soils Using Hybrid Machine Learning Technique.
Buildings 2023, 13, 255. MDPI.
"""

import os
import pypdf
import numpy as np
import pandas as pd

def classify_uscs(fines, ll, pi, sand, gravel):
    """Assigns geotechnical Unified Soil Classification System (USCS) domain."""
    if fines >= 50.0:
        if ll >= 50.0:
            return "CH (Fat Clay)" if pi >= (0.73 * (ll - 20)) else "MH (Elastic Silt)"
        else:
            return "CL (Lean Clay)" if (pi >= 7 and pi >= (0.73 * (ll - 20))) else "ML (Silt)"
    else:
        if sand >= gravel:
            return "SC (Clayey Sand)" if pi > 7 else "SM (Silty Sand)"
        else:
            return "GC (Clayey Gravel)" if pi > 7 else "GW/GP (Gravel)"

def extract_pdf_dataset(pdf_path="buildings-13-00255-v2.pdf"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Uploaded PDF not found at {pdf_path}")
        
    reader = pypdf.PdfReader(pdf_path)
    rows = []
    
    # Table A1 spans pages 15 to 18 (indices 14 to 17)
    for p in range(14, 18):
        text = reader.pages[p].extract_text()
        for line in text.split('\n'):
            parts = line.strip().split()
            # Row begins with serial integer 1..149 followed by 8 numerical properties
            if len(parts) >= 9 and parts[0].isdigit():
                try:
                    s_no = int(parts[0])
                    vals = [float(x) for x in parts[1:9]]
                    rows.append([s_no] + vals)
                except ValueError:
                    pass
                    
    df_raw = pd.DataFrame(
        rows,
        columns=['S_No', 'gravel_pct', 'sand_pct', 'fines_pct', 'LL', 'PL', 'OMC', 'MDD', 'CBR']
    )
    
    # Sort and remove duplicates if any
    df_raw = df_raw.drop_duplicates(subset=['S_No']).sort_values('S_No').reset_index(drop=True)
    
    # Compute Plasticity Index (PI = LL - PL)
    df_raw['PI'] = np.maximum(0.0, df_raw['LL'] - df_raw['PL'])
    
    # Metadata and soil classifications
    df_raw['source_study'] = "Nagaraju_et_al_2023_Lateritic_Soils"
    df_raw['geographical_region'] = "Tropical / Subtropical Laterite Subgrades"
    df_raw['test_condition'] = "Soaked 96-hr (ASTM D1883-16)"
    
    soil_domains = []
    for _, r in df_raw.iterrows():
        soil_domains.append(classify_uscs(r['fines_pct'], r['LL'], r['PI'], r['sand_pct'], r['gravel_pct']))
    df_raw['soil_domain'] = soil_domains
    
    # Standard column ordering
    col_order = [
        'S_No', 'source_study', 'geographical_region', 'test_condition', 'soil_domain',
        'LL', 'PL', 'PI', 'fines_pct', 'sand_pct', 'gravel_pct', 'MDD', 'OMC', 'CBR'
    ]
    df_final = df_raw[col_order]
    return df_final

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = extract_pdf_dataset("buildings-13-00255-v2.pdf")
    csv_path = os.path.join("data", "soil_cbr_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully extracted {len(df)} samples from uploaded 'buildings-13-00255-v2.pdf' into {csv_path}")
    print("\nStatistical Summary of Uploaded Dataset:")
    print(df[['LL', 'PL', 'PI', 'fines_pct', 'sand_pct', 'gravel_pct', 'MDD', 'OMC', 'CBR']].describe())
    print("\nSoil Domain Distribution:")
    print(df['soil_domain'].value_counts())
