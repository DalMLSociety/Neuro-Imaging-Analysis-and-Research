# analysis.py
import numpy as np
import pandas as pd
from scipy.stats import shapiro, pearsonr

def check_residuals_correlation(
    residuals: pd.Series, clinical: pd.Series
) -> tuple[float | None, float | None]:
    """
    Calculate Pearson correlation between residuals and a clinical score.

    Parameters:
        residuals (pd.Series): Residual values.
        clinical (pd.Series): Clinical score values.

    Returns:
        tuple: (correlation coefficient, p-value) or (None, None) if insufficient data.
    """
    mask = residuals.notna() & clinical.notna()
    if mask.sum() < 3:
        return None, None
    r, p = pearsonr(residuals[mask], clinical[mask])
    return r, p

def compare_variance(
    raw_fc: pd.Series, residuals: pd.Series
) -> tuple[float, float]:
    """
    Compare variance of raw FC and residuals.

    Parameters:
        raw_fc (pd.Series): Raw functional connectivity values.
        residuals (pd.Series): Residual values.

    Returns:
        tuple: (variance of raw_fc, variance of residuals)
    """
    return np.var(raw_fc.dropna(), ddof=1), np.var(residuals.dropna(), ddof=1)

def test_normality(residuals: pd.Series) -> tuple[float | None, float | None]:
    """
    Shapiro-Wilk test for normality of residuals.

    Parameters:
        residuals (pd.Series): Residual values.

    Returns:
        tuple: (test statistic, p-value) or (None, None) if insufficient data.
    """
    vals = residuals.dropna().values
    if len(vals) < 3:
        return None, None
    stat, p = shapiro(vals)
    return stat, p
