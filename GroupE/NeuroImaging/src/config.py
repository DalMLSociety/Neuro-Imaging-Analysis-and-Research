# config.py
import streamlit as st
import os, io, warnings, itertools, datetime as dt, re, math, random, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
#import moviepy

# --- Optional / Advanced libs ------------------------------------- #
opt = {}  #  Dictionary to keep track of library presence flags
try: import bct;                       opt['bct'] = True
except ImportError:                    opt['bct'] = False
try: import shap;                      opt['shap'] = True
except ImportError:                    opt['shap'] = False
try:
    import torch, torch.nn as nn, torch_geometric
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    opt['gnn'] = True
except ImportError:                    opt['gnn'] = False
try:
    import plotly.graph_objects as go
    opt['plotly'] = True
except ImportError:                    opt['plotly'] = False
try:
    import pywt; opt['pywt'] = True
except ImportError:                    opt['pywt'] = False
try:
    from prophet import Prophet; opt['prophet'] = True
except ImportError:                    opt['prophet'] = False
try:
    import moviepy.editor as mpy; opt['moviepy'] = True
except Exception as e:
    opt['moviepy'] = False
    st.warning(f"⚠️ MoviePy import error: {e}")
try:
    import statsmodels.api as sm; opt['statsmodels'] = True
except ImportError:
    opt['statsmodels'] = False

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Streamlit config
st.set_page_config(
    page_title="🧠 Neuroimaging Analysis Suite for fMRI Data",
    layout="wide", initial_sidebar_state="expanded"
)
st.title("🧠 Neuroimaging Analysis Suite for fMRI Data")
