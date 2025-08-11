# main.py
import streamlit as st
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import opt
from utils import load_clinical, random_sphere_coords
from features import calc_static_feats, calc_dfc, compute_plv
from visualization import plot_3d_graph, make_dfc_movie
from ml import svm_classifier, rf_regressor
from analysis import check_residuals_correlation, compare_variance, test_normality
if opt['gnn']:
    from gnn_models import WindowForecastDS, BrainGCN, DASTBlock, DASTForecaster
import os, pandas as pd, numpy as np, random, datetime as dt, warnings
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------ #
# 1) Sidebar – Setup & Config
# ------------------------------------------------------------------ #
st.sidebar.header("⚙️ 1. Setup & Configuration")

atlas = st.sidebar.selectbox("Atlas", ["aal3", "schaefer"])
data_dir = st.sidebar.text_input("📁 Base Time‑Series Folder", r"F:\dal\NIAR_Project_Zahra\FMRI\roi_time_series")

stat_opts = ['DegC', 'Clust']
if opt['bct']: stat_opts += ['Mods', 'GE']
stat_opts_show = {
    'DegC': 'Degree Centrality', 'Clust': 'Clustering Coefficient',
    'Mods': 'Modularity (Louvain)', 'GE': 'Global Efficiency'
}
sel_stats = st.sidebar.multiselect("Static Features", stat_opts, default=stat_opts)

# Dynamic FC config
st.sidebar.subheader("Dynamic FC")
do_dfc = st.sidebar.checkbox("🔄 Compute dFC / variance", value=True)
win = st.sidebar.slider("Window Size", 5, 50, 20, disabled=not do_dfc)
step = st.sidebar.slider("Step Size", 1, 20, 5, disabled=not do_dfc)

# Phase synchrony
st.sidebar.subheader("Phase Synchrony")
do_plv = st.sidebar.checkbox("⚡ Compute PLV", value=False, disabled=not opt['pywt'])

# Upload clinical + extra modalities
st.sidebar.header("📋 2. Upload Data")
clin_file = st.sidebar.file_uploader("Clinical CSV/XLSX", type=['csv','xlsx'])
extra_mod_file = st.sidebar.file_uploader("Extra Modality (behavioral / genetic) CSV", type=['csv'])

# Run button
run_btn = st.sidebar.button("🚀 Process Data")

# ------------------------------------------------------------------ #
# 2) Hidden Session-State Init
# ------------------------------------------------------------------ #
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
    st.session_state.df_full   = pd.DataFrame()
    st.session_state.dfc_raw   = {}  # subject → 3-D array
    st.session_state.dfc_tensor = None # Combined dFC tensor for DAST GCN
    st.session_state.dast_trained = False # Flag to check if DAST is trained

# ------------------------------------------------------------------ #
# 3) Data Loading & Feature Extraction
# ------------------------------------------------------------------ #
def process_all(base_path, atlas:str):
    paths = {g:os.path.join(base_path, atlas, g)
             for g in ('patient','control')}
    files = [os.path.join(p,f) for g,p in paths.items() if os.path.exists(p)
             for f in os.listdir(p) if f.endswith('.csv')]
    out_rows, dfc_dict = [], {}
    prog = st.progress(0, text="⏳ Processing time‑series…")
    for i, fp in enumerate(files):
        fn = os.path.basename(fp)
        m = re.search(r'(S\d{2,})', fn.upper())
        sid = m.group(1) if m else fn.split('_')[1].strip().upper()
        row = {'subject_id':sid, 'filepath':fp,
               'group':'patient' if 'patient' in fp.lower() else 'control'}
        feats_static = calc_static_feats(fp, sel_stats+['community_vec'])
        if feats_static: row.update(feats_static)
        if do_dfc:
            feats_dyn = calc_dfc(fp, win, step)
            if feats_dyn:
                row.update({k:v for k,v in feats_dyn.items() if k!='dFC_stack'})
                dfc_dict[sid] = feats_dyn['dFC_stack']
        if do_plv:
            ts = pd.read_csv(fp)
            plv_mat = compute_plv(ts)
            if plv_mat is not None:
                row['mean_plv'] = np.mean(plv_mat[np.triu_indices(plv_mat.shape[0],1)])
        out_rows.append(row)
        prog.progress((i+1)/len(files), text=f"Processed {fn} ({i+1}/{len(files)})")
    prog.empty()
    return pd.DataFrame(out_rows), dfc_dict

def compute_flexibility(dfc_stack, n_clusters=4):
    """
    Compute brain network flexibility for each ROI across dFC windows.
    Flexibility = fraction of times each ROI changes its cluster assignment.
    """
    from sklearn.cluster import KMeans
    if dfc_stack is None or len(dfc_stack) < 2:
        return None
    N = dfc_stack.shape[1]
    # For each window, cluster ROIs by their connectivity profile
    cluster_labels = []
    for m in dfc_stack:
        # Each ROI: its connectivity to others (row vector)
        X = m
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        cluster_labels.append(labels)
    cluster_labels = np.array(cluster_labels)  # shape: (n_windows, n_rois)
    # Flexibility: for each ROI, how often its cluster label changes
    flexibility = []
    for roi in range(N):
        roi_labels = cluster_labels[:, roi]
        changes = np.sum(roi_labels[1:] != roi_labels[:-1])
        flexibility.append(changes / (len(roi_labels) - 1))
    return np.array(flexibility)

if run_btn:
    if not data_dir or not os.path.exists(data_dir) or clin_file is None:
        st.error("❌ Both a valid base folder and clinical file are required.")
    else:
        clin_df = load_clinical(clin_file)
        if clin_df is None or 'subject_id' not in clin_df.columns:
            st.error("Clinical file must include 'subject_id'.")
        else:
            feat_df, dfc_dict = process_all(data_dir, atlas)
            if feat_df.empty:
                st.error("No features extracted – check folder structure.")
            else:
                clin_df['subject_id'] = clin_df['subject_id'].astype(str).str.upper().str.strip()
                feat_df['subject_id'] = feat_df['subject_id'].astype(str).str.upper().str.strip()
                # --- Compute flexibility for each subject if dFC is available ---
                flex_dict = {}
                if do_dfc and dfc_dict:
                    for sid, dfc_stack in dfc_dict.items():
                        flex = compute_flexibility(dfc_stack)
                        if flex is not None and isinstance(flex, np.ndarray) and flex.size > 0:
                            flex_dict[sid] = flex
                    # Add mean flexibility as a new feature to feat_df
                    feat_df['mean_flexibility'] = feat_df['subject_id'].map(
                        lambda sid: np.mean(flex_dict[sid]) if sid in flex_dict else np.nan
                    )
                    st.session_state.flexibility = flex_dict
                    # Debug: show how many subjects have flexibility
                    n_flex = sum(~np.isnan(feat_df['mean_flexibility']))
                    st.info(f"Flexibility computed for {n_flex} subjects.")
                    if n_flex == 0:
                        st.warning("No valid flexibility values computed. Check dFC window/step or data quality.")
                # Merge after flexibility is added
                df_merged = pd.merge(clin_df, feat_df, on='subject_id', how='outer')
                if extra_mod_file:
                    extra_df = load_clinical(extra_mod_file)
                    if extra_df is not None and 'subject_id' in extra_df.columns:
                        extra_df['subject_id'] = extra_df['subject_id'].astype(str).str.upper().str.strip()
                        df_merged = pd.merge(df_merged, extra_df, on='subject_id', how='left')
                st.session_state.df_full = df_merged
                st.session_state.dfc_raw = dfc_dict
                st.session_state.data_ready = True
                st.success(f"✅ Data ready! {df_merged.shape[0]} rows, {df_merged.shape[1]} columns.")
                st.session_state.dast_trained = False

# ------------------------------------------------------------------ #
# 4) Main Tabs – Basic & Advanced
# ------------------------------------------------------------------ #
if st.session_state.data_ready:
    df = st.session_state.df_full
    feature_cols = [c for c in df.columns if any(k in c for k in ['avg_','glob_eff','modularity','dFC_variance','mean_plv'])]
    basic_tabs = [
        "📄 Data", "📊 Static", "📈 Dynamic", "⏳ Longitudinal",
        "🤖 ML", "📤 Export"
    ]
    adv_tabs = [
        "🖥️ 3D Graph", "⚡ Phase", "📽️ dFC Movie", "📊 Feature Importance",
        "🧩 Clustering", "📈 Progression", "📑 Report"
    ]
    dast_tab = ["🧠 DAST GCN"] if opt['gnn'] and st.session_state.dfc_tensor is not None else []
    # Insert Flexibility tab after Dynamic FC (index 3)
    flex_tab_idx = 3
    # Swap Flexibility and Longitudinal tab order
    basic_tabs.insert(flex_tab_idx + 1, "⏳ Longitudinal")
    basic_tabs.insert(flex_tab_idx + 2, "🧬 Flexibility")
    # Remove the original Longitudinal and Flexibility if present
    if "⏳ Longitudinal" in basic_tabs[:flex_tab_idx + 1]:
        basic_tabs.remove("⏳ Longitudinal")
    if "🧬 Flexibility" in basic_tabs[:flex_tab_idx + 1]:
        basic_tabs.remove("🧬 Flexibility")
    tabs = st.tabs(basic_tabs + adv_tabs + dast_tab)

    # Data Preview
    with tabs[0]:
        st.header("👁️ Data Preview")
        st.dataframe(df)

    # Static FC
    with tabs[1]:
        st.header("Static FC Features")
        if feature_cols:
            sorted_feats = sorted(feature_cols)
            df_static = df[['group'] + sorted_feats]
            fig = sns.pairplot(df_static, hue='group', corner=True)
            st.pyplot(fig)

    # Dynamic FC
    with tabs[2]:
        st.header("Dynamic FC Variance")
        if 'dFC_variance' in df.columns:
            fig, ax = plt.subplots()
            sns.violinplot(x='group', y='dFC_variance', data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.info("dFC not computed.")

    # Longitudinal (now at index 3)
    with tabs[3]:
        st.header("Longitudinal Analysis")
        pt_df = df[df['group']=='patient'].copy()
        traj_feat = st.selectbox("Feature", ['sara','moca','mmse']+feature_cols)
        if traj_feat in pt_df.columns and 'age' in pt_df.columns:
            fig, ax = plt.subplots()
            sns.lineplot(data=pt_df, x='age', y=traj_feat,
                         hue='subject_id', legend=None, ax=ax)
            st.pyplot(fig)

    # Flexibility Tab (now at index 4)
    with tabs[4]:
        st.header("🧬 Brain Network Flexibility")
        if 'mean_flexibility' in df.columns:
            st.write("Mean flexibility: Fraction of times each ROI changes its network membership over time (based on dFC).")
            fig, ax = plt.subplots()
            sns.boxplot(x='group', y='mean_flexibility', data=df, ax=ax)
            st.pyplot(fig)
            st.dataframe(df[['subject_id', 'group', 'mean_flexibility']])
            # Optional: Show flexibility profile for a selected subject
            if hasattr(st.session_state, 'flexibility') and st.session_state.flexibility:
                sid_flex = st.selectbox("Subject (Flexibility Profile)", list(st.session_state.flexibility.keys()), key='flex_profile_subject')
                flex_vec = st.session_state.flexibility[sid_flex]
                fig2, ax2 = plt.subplots()
                ax2.plot(flex_vec, marker='o')
                ax2.set_xlabel("ROI")
                ax2.set_ylabel("Flexibility")
                ax2.set_title(f"Flexibility Profile for {sid_flex}")
                st.pyplot(fig2)
        else:
            st.info("Flexibility not computed. Run dFC analysis first.")

    # Machine Learning
    with tabs[5]:
        st.header("Machine Learning")
        st.subheader("SVM Classifier (Group-aware)")
        if feature_cols and df['group'].nunique() == 2 and 'subject_id' in df.columns:
            X = df[feature_cols]
            y = df['group']
            groups = df['subject_id']
            C = st.slider("C (SVM)", 0.01, 10.0, 1.0)
            pipe = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', StandardScaler()),
                ('svm', SVC(C=C, kernel='rbf', random_state=42))
            ])
            gkf = GroupKFold(n_splits=3)
            acc = cross_val_score(pipe, X, y, cv=gkf, groups=groups).mean()
            st.metric("CV Accuracy (GroupKFold)", f"{acc:.2%}")
        else:
            st.warning("Missing subject_id or class labels for group-aware validation.")
        st.subheader("Random Forest – SARA (Group-aware)")
        if 'sara' in df.columns and feature_cols and 'subject_id' in df.columns:
            pt = df[df['group'] == 'patient'].dropna(subset=['sara'])
            if not pt.empty:
                X = pt[feature_cols]
                y = pt['sara']
                groups = pt['subject_id']
                n_tree = st.slider("Trees", 10, 200, 100)
                pipe = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(n_estimators=n_tree, random_state=42))
                ])
                gkf = GroupKFold(n_splits=3)
                mse = -cross_val_score(pipe, X, y, cv=gkf, groups=groups,
                                    scoring='neg_mean_squared_error').mean()
                st.metric("MSE (GroupKFold)", f"{mse:.2f}")
                if opt['shap'] and st.checkbox("🔍 Show SHAP", key='rf_shap_sara'):
                    pipe.fit(X, y)
                    expl = shap.Explainer(pipe.named_steps['rf'])
                    X_transformed = pipe.named_steps['scaler'].transform(
                        pipe.named_steps['imputer'].transform(X))
                    shap_vals = expl(X_transformed)
                    shap.summary_plot(shap_vals, X, feature_names=feature_cols, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig, bbox_inches='tight'); plt.clf()
        else:
            st.warning("Missing subject_id or class labels for group-aware validation.")

    # Export
    with tabs[6]:
        st.header("Export")
        st.download_button("CSV", df.to_csv(index=False).encode(),
                           "data_processed.csv", "text/csv")

    # 3D Graph
    with tabs[7]:
        st.header("3D Brain Graph")
        sid = st.selectbox("Subject", df['subject_id'].dropna().unique(), key='3d_graph_subject')
        row_3d = df[df['subject_id']==sid].iloc[0]
        if os.path.exists(row_3d['filepath']) and opt['plotly']:
            ts = pd.read_csv(row_3d['filepath'])
            conn = ts.corr().fillna(0).values
            comm_vec = row_3d.get('community_vec', None)
            thresh = st.slider("Edge threshold", 0.0, 1.0, .3, .05, key='3d_graph_thresh')
            plot_3d_graph(conn, comm_vec, thresh)
        else:
            st.info("Cannot render graph (missing file or plotly).")

    # Phase Synchrony
    with tabs[8]:
        st.header("Phase Synchrony (PLV)")
        if 'mean_plv' in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='group', y='mean_plv', data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.info("PLV not computed.")

    # dFC Movie
    with tabs[9]:
        st.header("dFC Animation")
        subjects_with_dfc = [sid for sid, tensor in st.session_state.dfc_raw.items() if tensor is not None and tensor.shape[0] > 0]
        if subjects_with_dfc:
            sid_movie = st.selectbox("Subject (dFC Movie)", subjects_with_dfc, key='dfc_movie_subject')
            if st.button("🎬 Generate Movie", key='generate_movie_btn'):
                with st.spinner(f"Generating movie for {sid_movie}..."):
                    vid_bytes = make_dfc_movie(st.session_state.dfc_raw[sid_movie])
                if vid_bytes:
                    st.video(vid_bytes, format="video/mp4")
                    st.download_button("Download mp4", vid_bytes,
                                       f"{sid_movie}_dfc.mp4", "video/mp4")
        else:
             st.info("No dFC data available to generate movie.")

    # Feature Importance
    with tabs[10]:
        st.header("Feature Importance (SHAP)")
        if not opt.get('shap', False):
            st.info("SHAP library not available.")
        elif 'sara' not in df.columns or not feature_cols or 'subject_id' not in df.columns:
            st.info("SARA column, features, or subject_id missing for SHAP analysis.")
        else:
            pt = df[df['group'] == 'patient'].dropna(subset=['sara'])
            if pt.empty:
                st.info("No patient data with SARA scores available for SHAP analysis.")
            else:
                X = pt[feature_cols]
                y = pt['sara']
                n_tree = st.slider("Trees (for SHAP RF)", 10, 200, 100, key='shap_rf_trees')
                if st.checkbox("🔍 Show SHAP Summary Plot", key='show_shap_summary'):
                    from sklearn.pipeline import Pipeline
                    from sklearn.impute import SimpleImputer
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.ensemble import RandomForestRegressor
                    import shap
                    pipe = Pipeline([
                        ('imputer', SimpleImputer()),
                        ('scaler', StandardScaler()),
                        ('rf', RandomForestRegressor(n_estimators=n_tree, random_state=42))
                    ])
                    pipe.fit(X, y)
                    expl = shap.Explainer(pipe.named_steps['rf'])
                    X_transformed = pipe.named_steps['scaler'].transform(
                        pipe.named_steps['imputer'].transform(X))
                    shap_vals = expl(X_transformed)
                    shap.summary_plot(shap_vals, X, feature_names=feature_cols, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig, bbox_inches='tight')
                    plt.clf()
                else:
                    st.info("Check the box above to compute and display SHAP feature importance for SARA prediction.")

    # Clustering / Prognosis
    with tabs[11]:
        st.header("Unsupervised Clustering")
        if feature_cols:
            X = df[feature_cols].fillna(df[feature_cols].mean())
            k = st.slider("Clusters", 2, 6, 3, 1, key='kmeans_clusters')
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
            df['cluster'] = km.labels_
            # --- PCA error handling ---
            if X.shape[0] >= 2 and X.shape[1] >= 2:
                pca = PCA(n_components=2).fit_transform(X)
                fig, ax = plt.subplots()
                palette = sns.color_palette("Set1", n_colors=k)
                sns.scatterplot(
                    x=pca[:, 0], y=pca[:, 1],
                    hue=df['cluster'],
                    style=df['group'],
                    palette=palette,
                    s=80,
                    ax=ax
                )
                ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
                st.pyplot(fig)
                st.dataframe(df[['subject_id','cluster','sara','group']])
            else:
                st.warning("Not enough samples or features for PCA clustering. At least 2 samples and 2 features are required.")
        else:
            st.info("No features available for clustering.")

    # Disease Progression
    with tabs[12]:
        st.header("Disease Progression Forecast")
        # --- New: check for group column and patient label ---
        if 'group' not in df.columns:
            st.warning("Missing 'group' column. Cannot determine which subjects are patients for disease progression analysis.")
        elif 'age' in df.columns and 'sara' in df.columns and opt['prophet']:
            patient_sara_counts = df[df['group']=='patient'].dropna(subset=['age','sara']).groupby('subject_id').size()
            patients_with_enough_data = patient_sara_counts[patient_sara_counts >= 2].index.tolist()
            if patients_with_enough_data:
                sid_rep = st.selectbox("Patient for Forecast", patients_with_enough_data, key='prophet_subject')
                sub = df[df['subject_id']==sid_rep].dropna(subset=['age','sara']).copy()
                if len(sub) >= 2:
                    base_date = dt.date(1970,1,1)
                    sub_prophet = pd.DataFrame({
                        'ds':[base_date + dt.timedelta(days=int(a*365.25)) for a in sub['age']],
                        'y':sub['sara']
                    })
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = Prophet()
                        m.fit(sub_prophet)
                        future = m.make_future_dataframe(periods=3, freq='Y')
                        fcst = m.predict(future)
                    fig2 = m.plot(fcst)
                    st.pyplot(fig2)
                else:
                     st.info(f"Subject {sid_rep} does not have enough data points (at least 2) for Prophet forecast.")
            else:
                 st.info("No patients with enough age and SARA data points for Prophet forecast.")
        else:
            if not opt['prophet']:
                 st.info("Prophet library not available.")
            elif 'age' not in df.columns or 'sara' not in df.columns:
                st.info("Missing 'age' or 'sara' column for disease progression forecast.")
            else:
                st.info("Disease progression analysis could not be performed. Please check your data and configuration.")

    # Clinical Report
    with tabs[13]:
        st.header("Personalized Clinical Report")
        if not df.empty:
            sid_rep = st.selectbox("Subject (Report)", df['subject_id'].dropna().unique(), key='clinical_report_subject')
            subj = df[df['subject_id']==sid_rep].iloc[0]
            st.subheader("Demographics & Clinical")
            demo_cols = ['subject_id','group','age','gender', 'sara', 'moca', 'mmse']
            st.json({c:str(subj[c]) for c in demo_cols if c in subj and pd.notna(subj[c])})
            st.subheader("Key Graph Metrics")
            graph_metrics_present = {c:round(subj[c],3) for c in feature_cols if c in subj and pd.notna(subj[c])}
            if graph_metrics_present:
                st.json(graph_metrics_present)
            else:
                st.info("No key graph metrics computed for this subject.")
            # Ataxia stage classification
            if 'sara' in subj and pd.notna(subj['sara']):
                sara_score = subj['sara']
                if sara_score <= 7:
                    stage = "Early stage ataxia"
                elif sara_score <= 15:
                    stage = "Moderate stage ataxia"
                else:
                    stage = "Advanced stage ataxia"
                st.write(f"🩺 Disease stage: **{stage}** (SARA={sara_score})")
            if 'cluster' in subj and pd.notna(subj['cluster']):
                st.write(f"🔎 Cluster assignment: **{subj['cluster']}**")
            else:
                st.info("Clustering not performed or no cluster assigned for this subject.")
        else:
            st.info("No data loaded yet to generate a clinical report.")

    # DAST GCN Forecasting
    if opt['gnn'] and st.session_state.dfc_tensor is not None:
        with tabs[len(basic_tabs) + len(adv_tabs)]:
            st.header("🧠 DAST GCN Forecasting")
            st.info("Forecast the next dynamic FC matrix using a DAST GCN model.")
            if st.session_state.dfc_tensor is not None and st.session_state.dfc_tensor.shape[0] > 0:
                W, N, _ = st.session_state.dfc_tensor.shape
                st.subheader("Model Parameters")
                SEQ_LEN_IN = st.slider("Input Sequence Length", 2, min(20, W-2), 5, key='dast_seq_len')
                DAST_D = st.slider("DAST Hidden Dim (d)", 16, 128, 64, key='dast_d')
                DAST_K = st.slider("DAST Blocks (k)", 1, 5, 2, key='dast_k')
                EPOCHS = st.slider("Epochs", 1, 50, 10, key='dast_epochs')
                BATCH = st.slider("Batch Size", 8, 64, 32, key='dast_batch')
                LR = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f", key='dast_lr')
                VAL_RATIO = st.slider("Validation Split Ratio", 0.1, 0.5, 0.2, key='dast_val_ratio')
                PATIENCE = st.slider("Early Stopping Patience", 1, 20, 5, key='dast_patience')
                if st.button("🏋️ Train DAST GCN Model", key='train_dast_btn'):
                    if W < SEQ_LEN_IN + 2:
                         st.warning(f"Not enough dynamic FC windows ({W}) for training with sequence length {SEQ_LEN_IN}.")
                    else:
                        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        st.write(f"Using device: {DEVICE}")
                        train_size_total = W - SEQ_LEN_IN - 1
                        val_size = int(train_size_total * VAL_RATIO)
                        train_size = train_size_total - val_size
                        if train_size <= 0 or val_size <= 0:
                             st.warning(f"Data split resulted in zero or negative training ({train_size}) or validation ({val_size}) samples. Adjust parameters.")
                        else:
                            train_indices_end = train_size + SEQ_LEN_IN
                            train_data_for_scaler = st.session_state.dfc_tensor[:train_indices_end].reshape(train_indices_end, -1)
                            scaler = StandardScaler()
                            scaler.fit(train_data_for_scaler)
                            dfc_norm_tensor = scaler.transform(st.session_state.dfc_tensor.reshape(W, -1)).reshape(st.session_state.dfc_tensor.shape)
                            train_dataset = WindowForecastDS(dfc_norm_tensor[:train_size + SEQ_LEN_IN], SEQ_LEN_IN)
                            val_dataset = WindowForecastDS(dfc_norm_tensor[train_size:train_size+val_size + SEQ_LEN_IN], SEQ_LEN_IN)
                            full_dataset = WindowForecastDS(dfc_norm_tensor, SEQ_LEN_IN)
                            train_dl = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
                            val_dl = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)
                            full_dl = DataLoader(full_dataset, batch_size=BATCH, shuffle=False)
                            model = DASTForecaster(n_roi=N, d=DAST_D, k=DAST_K).to(DEVICE)
                            opt_dast   = torch.optim.Adam(model.parameters(), lr=LR)
                            mse_loss   = nn.MSELoss()
                            best_val_loss = float('inf')
                            epochs_no_improve = 0
                            train_losses, val_losses = [], []
                            st.subheader("Training Progress")
                            progress_bar = st.progress(0)
                            loss_chart = st.line_chart()
                            for epoch in range(1, EPOCHS+1):
                                model.train()
                                running_train = 0
                                for xb, yb in train_dl:
                                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                                    opt_dast.zero_grad()
                                    loss = mse_loss(model(xb), yb)
                                    loss.backward()
                                    opt_dast.step()
                                    running_train += loss.item() * xb.size(0)
                                train_mse = running_train / len(train_dl.dataset) / N**2
                                train_losses.append(train_mse)
                                model.eval()
                                running_val = 0
                                with torch.no_grad():
                                    for xb, yb in val_dl:
                                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                                        val_pred = model(xb)
                                        loss_val = mse_loss(val_pred, yb)
                                        running_val += loss_val.item() * xb.size(0)
                                val_mse = running_val / len(val_dl.dataset) / N**2
                                val_losses.append(val_mse)
                                progress_bar.progress(epoch / EPOCHS)
                                loss_chart.add_rows([[train_mse, val_mse]])
                                st.write(f"Epoch {epoch:02d} | Train MSE/edge = {train_mse:.6f} | Validation MSE/edge = {val_mse:.6f}")
                                if val_mse < best_val_loss:
                                    best_val_loss = val_mse
                                    epochs_no_improve = 0
                                    torch.save(model.state_dict(), "best_dast_forecaster.pt")
                                    st.info("Validation loss improved, model state saved.")
                                else:
                                    epochs_no_improve += 1
                                    st.warning(f"No improvement in validation loss for {epochs_no_improve} epochs.")
                                    if epochs_no_improve >= PATIENCE:
                                        st.error("Early stopping triggered.")
                                        break
                            st.success("DAST GCN Training Complete.")
                            st.session_state.dast_trained = True
                            st.session_state.dast_scaler = scaler
                            st.session_state.dast_seq_len = SEQ_LEN_IN
                if st.session_state.dast_trained:
                    st.subheader("DAST GCN Prediction")
                    if W < st.session_state.dast_seq_len + 1:
                         st.warning("Not enough dynamic FC windows for prediction.")
                    else:
                        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        _, N, _ = st.session_state.dfc_tensor.shape
                        try:
                            model = DASTForecaster(n_roi=N, d=DAST_D, k=DAST_K).to(DEVICE)
                            model.load_state_dict(torch.load("best_dast_forecaster.pt", map_location=DEVICE))
                            model.eval()
                            st.success("Trained model loaded.")
                            scaler = st.session_state.dast_scaler
                            SEQ_LEN_IN = st.session_state.dast_seq_len
                            full_normalized_tensor = scaler.transform(st.session_state.dfc_tensor.reshape(W, -1)).reshape(st.session_state.dfc_tensor.shape)
                            test_dataset_full = WindowForecastDS(full_normalized_tensor, SEQ_LEN_IN)
                            num_test_samples = max(1, int(len(test_dataset_full) * 0.2))
                            test_start_index_in_dataset = len(test_dataset_full) - num_test_samples
                            test_dataloader = DataLoader(torch.utils.data.Subset(test_dataset_full, range(test_start_index_in_dataset, len(test_dataset_full))),
                                                         batch_size=BATCH, shuffle=False)
                            preds = []
                            targets = []
                            errors = []
                            st.subheader("Prediction Results")
                            prediction_progress = st.progress(0)
                            with torch.no_grad():
                                for i, (xb, yb) in enumerate(test_dataloader):
                                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                                    pred = model(xb)
                                    batch_errors = torch.sqrt(mse_loss(pred, yb) / N**2).cpu().numpy().tolist()
                                    errors.extend(batch_errors)
                                    preds.extend(pred.cpu().numpy())
                                    targets.extend(yb.cpu().numpy())
                                    prediction_progress.progress((i + 1) / len(test_dataloader))
                            st.success("Prediction Complete.")
                            if errors:
                                avg_rmse = np.mean(errors)
                                st.write(f"Average RMSE/edge over test windows: **{avg_rmse:.6f}**")
                                st.subheader("Example Prediction vs. Ground Truth")
                                if preds and targets:
                                    example_idx = random.randint(0, len(preds) - 1)
                                    pred_example = preds[example_idx]
                                    target_example = targets[example_idx]
                                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                                    sns.heatmap(pred_example, cmap='coolwarm', center=0, ax=axes[0])
                                    axes[0].set_title("Predicted dFC Matrix")
                                    sns.heatmap(target_example, cmap='coolwarm', center=0, ax=axes[1])
                                    axes[1].set_title("Ground Truth dFC Matrix")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                        except FileNotFoundError:
                            st.warning("No trained model found. Please train the model first.")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                else:
                    st.info("Train the DAST GCN model first to see prediction results.")
            else:
                st.info("No dynamic FC data available for DAST GCN analysis.")

    # --- 1️⃣ Statistical checks on the residuals ---
    if st.session_state.data_ready and 'residuals' in df.columns:
        st.subheader('Statistical Checks on Residuals')
        import matplotlib.pyplot as plt
        import seaborn as sns
        if opt.get('statsmodels', False):
            import statsmodels.api as sm
        clinical_vars = ['sara', 'mmse', 'moca', 'age', 'cag']
        for var in clinical_vars:
            if var in df.columns:
                r, p = check_residuals_correlation(df['residuals'], df[var])
                st.write(f'Correlation (residuals vs {var.upper()}): r={r:.3f}, p={p:.3g}' if r is not None else f'Not enough data for {var}.')
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[var], y=df['residuals'], ax=ax)
                ax.set_title(f'Residuals vs {var.upper()}')
                st.pyplot(fig)
        if 'raw_fc' in df.columns:
            v_raw, v_res = compare_variance(df['raw_fc'], df['residuals'])
            st.write(f'Variance: raw FC={v_raw:.4f}, residuals={v_res:.4f}')
        # Histogram and Q-Q plot
        fig, axes = plt.subplots(1,2,figsize=(8,3))
        sns.histplot(df['residuals'].dropna(), kde=True, ax=axes[0])
        axes[0].set_title('Residuals Histogram')
        if opt.get('statsmodels', False):
            sm.qqplot(df['residuals'].dropna(), line='s', ax=axes[1])
            axes[1].set_title('Q-Q Plot')
        st.pyplot(fig)
        stat, p = test_normality(df['residuals'])
        st.write(f'Shapiro-Wilk normality test: stat={stat:.3f}, p={p:.3g}' if stat is not None else 'Not enough data for normality test.')

    # --- 2️⃣ Run the same models on raw FC vs residual FC ---
    if st.session_state.data_ready and 'raw_fc' in df.columns and 'residuals' in df.columns and 'group' in df.columns:
        st.subheader('ML Comparison: Raw FC vs Residual FC')
        feature_cols = [c for c in df.columns if 'fc' in c or 'residual' in c]
        X_raw = df[[c for c in feature_cols if 'raw_fc' in c]]
        X_resid = df[[c for c in feature_cols if 'residual' in c]]
        y = df['group']
        groups = df['subject_id'] if 'subject_id' in df.columns else None
        from ml import svm_classifier, rf_regressor, run_ml_models
        svm_res = run_ml_models(X_raw, X_resid, y, groups, svm_classifier)
        rf_res = run_ml_models(X_raw, X_resid, y, groups, rf_regressor)
        import pandas as pd
        ml_df = pd.DataFrame({
            'Model': ['SVM', 'Random Forest'],
            'Raw FC': [svm_res['raw'], rf_res['raw']],
            'Residual FC': [svm_res['residual'], rf_res['residual']]
        })
        st.dataframe(ml_df)
        fig, ax = plt.subplots()
        ml_df.set_index('Model').plot(kind='bar', ax=ax)
        ax.set_ylabel('Score (Accuracy or -MSE)')
        st.pyplot(fig)

    # --- 3️⃣ Clinical interpretation of predictions ---
    if st.session_state.data_ready and 'prediction' in df.columns and 'month' in df.columns:
        st.subheader('Clinical Interpretation of Predictions')
        for sid, subdf in df.groupby('subject_id'):
            st.write(f'Patient {sid}:')
            pred_traj = subdf.sort_values('month')[['month','prediction']]
            fig, ax = plt.subplots()
            ax.plot(pred_traj['month'], pred_traj['prediction'], marker='o')
            ax.set_xlabel('Month'); ax.set_ylabel('Prediction')
            ax.set_title(f'Prediction Trajectory for {sid}')
            st.pyplot(fig)
            if pred_traj['prediction'].diff().min() < -0.1:
                st.write('  ⚠️ FC drop detected. Possible motor decline or worsening condition.')
            else:
                st.write('  No significant FC drop detected.')
            note = st.text_area(f'Clinical note for {sid}', '', key=f'note_{sid}')

    # --- 4️⃣ Include clinical variables as features in the model ---
    clinical_vars = ['age', 'gender', 'sara', 'mmse', 'cag']
    clinical_feats = [c for c in clinical_vars if c in df.columns]
    if clinical_feats:
        st.subheader('Add Clinical Variables as Features')
        selected_clin = st.multiselect('Select clinical variables to include:', clinical_feats, default=clinical_feats)
        X_clinical = df[selected_clin]
        st.write('Selected clinical features will be used in the next ML run.')
        # Example: concatenate with FC features for ML
        # X_full = pd.concat([X_raw, X_clinical], axis=1)

    # --- 5️⃣ Compare our results with similar studies ---
    st.subheader('Comparison with Similar Studies')
    st.write('Enter metrics from similar studies for comparison:')
    studies = []
    for i in range(1, 4):
        s = st.text_input(f'Study {i} (e.g., RMSE=0.25, N=30, Ref=Smith2020)')
        if s: studies.append(s)
    our_rmse = rf_res['residual'] if 'rf_res' in locals() else None
    # Parse RMSE values, handle errors robustly
    rmse_vals = []
    for s in studies:
        try:
            # Try to extract RMSE value (float after 'RMSE=' or 'rmse=')
            m = re.search(r'RMSE\s*=\s*([0-9.]+)', s, re.IGNORECASE)
            if m:
                rmse_vals.append(float(m.group(1)))
            else:
                rmse_vals.append(None)
        except Exception:
            rmse_vals.append(None)
    comp_df = pd.DataFrame({
        'Study': ['Ours'] + [f'Study {i}' for i in range(1, len(studies)+1)],
        'RMSE': [our_rmse] + rmse_vals
    })
    # Only keep rows with valid numeric RMSE for plotting
    comp_df_plot = comp_df[pd.to_numeric(comp_df['RMSE'], errors='coerce').notnull()]
    fig, ax = plt.subplots()
    if not comp_df_plot.empty:
        comp_df_plot.set_index('Study')['RMSE'].plot(kind='bar', ax=ax)
        ax.set_ylabel('RMSE')
        st.pyplot(fig)
    else:
        st.info("No valid numeric RMSE values to plot.")
def random_sphere_coords(n:int, seed:int=42):
    """Generate pseudo‑random but reproducible 3‑D coordinates on a sphere."""
    rng = np.random.RandomState(seed)
    phi = np.arccos(1 - 2*rng.rand(n))
    theta = 2*np.pi*rng.rand(n)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.vstack([x,y,z]).T
