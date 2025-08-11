import os
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)


# --- Converting temporal data into FC (correlation) matrix and extracting edges
def extract_edges(df):
    conn = df.corr().values
    triu_indices = np.triu_indices_from(conn, k=1)
    edge_values = conn[triu_indices]
    edge_cols = [f"Edge_{i}" for i in range(len(edge_values))]
    return pd.DataFrame([edge_values], columns=edge_cols)

# --- Fit the LME model for each edge
def fit_lme_model(df):
    results = {}
    #for edge in [col for col in df.columns if col.startswith("Edge_")]:
    for edge in [col for col in df.columns if col.startswith("Edge_")][:20]:  

        try:
            model = smf.mixedlm(f"{edge} ~ Age + Group", df, groups=df["Subject_ID"])
            result = model.fit()
            results[edge] = result
        except Exception as e:
            print(f"⚠️ Error fitting model for {edge}: {e}")
            results[edge] = None
    return results

# --- Taking residuals
def get_residuals(df, model_results):
    residuals_df = pd.DataFrame(index=df.index)
    for edge, result in model_results.items():
        if result is not None:
            try:
                pred = result.fittedvalues
                residuals_df[edge] = df[edge] - pred
            except Exception as e:
                print(f"⚠️ Could not compute residuals for {edge}: {e}")
                residuals_df[edge] = pd.NA
        else:
            residuals_df[edge] = pd.NA
    return residuals_df

# --------------------------- main implementation -----------------------------
if __name__ == "__main__":
    control_files = glob.glob("roi_time_series/aal3/control/*.csv")
    patient_files = glob.glob("roi_time_series/aal3/patient/*.csv")
    all_files = control_files + patient_files

    all_dfs = []
    for file in all_files:
        subject_id = os.path.basename(file).replace(".csv", "")
        raw_df = pd.read_csv(file)

        edge_df = extract_edges(raw_df)  # Calculation of FC matrix → Edge
        edge_df["Subject_ID"] = subject_id
        edge_df["Age"] = 65 if "control" in file else 58   
        edge_df["Group"] = 0 if "control" in file else 1   # 0 = control, 1 = patient

        all_dfs.append(edge_df)

    df = pd.concat(all_dfs, ignore_index=True)

    print("🔍 Starting LME modeling...")
    model_results = fit_lme_model(df)

    print("📉 Calculating residuals...")
    residuals_df = get_residuals(df, model_results)

    print("Residuals shape:", residuals_df.shape)


    os.makedirs("results", exist_ok=True)
    residuals_df.to_csv("results/residual_ts.csv", index=False)
    print("✅ Residuals saved to results/residual_ts.csv")

print(f"✅ Total successful models: {sum([r is not None for r in model_results.values()])} / {len(model_results)}")
