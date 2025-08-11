# utils.py
import os
import pandas as pd
import streamlit as st
import numpy as np
import re

def load_clinical(f):
    """Read clinical table (latin1 fallback) → DataFrame."""
    try:
        if f.name.endswith('.csv'): return pd.read_csv(f, encoding='latin1')
        return pd.read_excel(f, engine='openpyxl')
    except Exception as e:
        st.error(f"Clinical file read error: {e}"); return None

def random_sphere_coords(n:int, seed:int=42):
    """Generate pseudo‑random but reproducible 3‑D coordinates on a sphere."""
    rng = np.random.RandomState(seed)
    phi = np.arccos(1 - 2*rng.rand(n))
    theta = 2*np.pi*rng.rand(n)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.vstack([x,y,z]).T

def compute_flexibility(dfc_stack, n_clusters=4):
    """
    Compute brain network flexibility for each ROI across dFC windows.
    Flexibility = fraction of times each ROI changes its cluster assignment.
    """
    from sklearn.cluster import KMeans
    if dfc_stack is None or len(dfc_stack) < 2:
        return None
    N = dfc_stack.shape[1]
    cluster_labels = []
    for m in dfc_stack:
        X = m
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        cluster_labels.append(labels)
    cluster_labels = np.array(cluster_labels)
    flexibility = []
    for roi in range(N):
        roi_labels = cluster_labels[:, roi]
        changes = np.sum(roi_labels[1:] != roi_labels[:-1])
        flexibility.append(changes / (len(roi_labels) - 1))
    return np.array(flexibility)

# You can add more utility functions here as needed for the project.
