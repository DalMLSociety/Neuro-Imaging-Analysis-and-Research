# ml.py
import pandas as pd
import streamlit as st
from config import opt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

def svm_classifier(X, y, groups, C=1.0):
    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=C, kernel='rbf', random_state=42))
    ])
    gkf = GroupKFold(n_splits=3)
    acc = cross_val_score(pipe, X, y, cv=gkf, groups=groups).mean()
    return acc

def rf_regressor(X, y, groups, n_tree=100):
    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=n_tree, random_state=42))
    ])
    gkf = GroupKFold(n_splits=3)
    mse = -cross_val_score(pipe, X, y, cv=gkf, groups=groups,
                        scoring='neg_mean_squared_error').mean()
    return mse

def run_ml_models(X_raw, X_resid, y, groups, model_func):
    """Run the same ML model on raw and residual FC, return metrics for both."""
    res = {}
    res['raw'] = model_func(X_raw, y, groups)
    res['residual'] = model_func(X_resid, y, groups)
    return res
