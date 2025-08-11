# features.py
import os, warnings
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
from config import opt

@st.cache_data(show_spinner="📊 Calculation of static characteristics...")
def calc_static_feats(ts_path:str, feats:list):
    if not os.path.exists(ts_path): return None
    try:
        ts = pd.read_csv(ts_path)
        if ts.shape[1] < 2: return None
        conn = ts.corr().fillna(0).values
        G = nx.from_numpy_array(np.abs(conn))
        res = {}
        if 'DegC' in feats:
            res['avg_deg_cent'] = np.mean(list(nx.degree_centrality(G).values()))
        if 'Clust' in feats:
            res['avg_clust'] = nx.average_clustering(G)
        if opt['bct']:
            if 'Mods' in feats:
                _, q = bct.modularity_louvain_und(np.abs(conn))
                res['modularity'] = q
            if 'GE' in feats:
                res['glob_eff'] = bct.efficiency_wei(np.abs(conn))
        if 'community_vec' in feats:
            if opt['bct']:
                comm, _ = bct.modularity_louvain_und(np.abs(conn))
            else:
                comm_iter = nx.algorithms.community.greedy_modularity_communities(G)
                comm = np.zeros(len(G), dtype=int)
                for cidx, nodes in enumerate(comm_iter):
                    for n in nodes: comm[n] = cidx
            res['community_vec'] = comm
        return res
    except Exception as e:
        warnings.warn(str(e)); return None

@st.cache_data(show_spinner="⚡ Dynamic FC calculation...")
def calc_dfc(ts_path:str, win:int, step:int):
    if not os.path.exists(ts_path): return None
    try:
        ts = pd.read_csv(ts_path)
        T, N = ts.shape
        if T < win: return None
        mats = []
        for s in range(0, T-win+1, step):
            window = ts.iloc[s:s+win]
            mats.append(window.corr().fillna(0).values)
        dFC = np.array(mats)
        var_conn = np.var(dFC, axis=0)
        mean_var = np.mean(var_conn[np.triu_indices(N, 1)])
        return {'dFC_variance': mean_var, 'dFC_stack': dFC}
    except Exception as e:
        warnings.warn(str(e)); return None

def compute_plv(ts:pd.DataFrame, fs:int=1):
    if not opt['pywt']:
        st.warning("PyWavelets not installed ➜ Phase synchrony is disabled"); return None
    from scipy.signal import hilbert
    phases = np.angle(hilbert(ts.values, axis=0))
    N = phases.shape[1]
    plv_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            diff = phases[:,i]-phases[:,j]
            plv = abs(np.mean(np.exp(1j*diff)))
            plv_mat[i,j] = plv_mat[j,i] = plv
    return plv_mat

# ---------------------------- ✅ NEW FUNCTION ----------------------------

def run_sliding_window_on_all(input_paths: dict, output_dir="results", win=30, step=5):
    """
    Run calc_dfc on multiple input CSVs (e.g., raw and residual) and save results.
    Example input:
        input_paths = {
            "raw": "data/ts.csv",
            "resid": "results/residual_ts.csv"
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    for label, path in input_paths.items():
        try:
            print(f"⏳ Processing {label} → {path}")
            dfc_result = calc_dfc(path, win=win, step=step)
            if dfc_result and "dFC_stack" in dfc_result:
                out_path = os.path.join(output_dir, f"dfc_{label}_stack.npy")
                np.save(out_path, dfc_result["dFC_stack"])
                print(f"✅ Saved → {out_path}")
            else:
                print(f"⚠️ Could not calculate dFC for {label}")
        except Exception as e:
            print(f"❌ Error in {label}: {e}")

run_sliding_window_on_all({
    "resid": "results/residual_ts.csv"
})
