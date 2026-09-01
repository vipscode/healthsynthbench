"""
Fidelity metrics: how statistically similar is synthetic data to real data?
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def marginal_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """Mean (1 - KS statistic) across columns. 1.0 = identical marginals."""
    scores = {}
    for col in real_df.columns:
        stat, _ = ks_2samp(real_df[col], synth_df[col])
        scores[col] = 1 - stat
    return scores, float(np.mean(list(scores.values())))


def correlation_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """1 - normalized Frobenius distance between correlation matrices."""
    real_corr = real_df.corr().values
    synth_corr = synth_df.corr().values
    diff = np.linalg.norm(real_corr - synth_corr)
    max_diff = np.linalg.norm(np.ones_like(real_corr) * 2)  # worst case bound
    return float(1 - diff / max_diff)


def fidelity_report(real_df, synth_df):
    col_scores, mean_marginal = marginal_fidelity(real_df, synth_df)
    corr_score = correlation_fidelity(real_df, synth_df)
    overall = 0.6 * mean_marginal + 0.4 * corr_score
    return {
        "per_column_marginal_fidelity": col_scores,
        "mean_marginal_fidelity": round(mean_marginal, 4),
        "correlation_fidelity": round(corr_score, 4),
        "overall_fidelity_score": round(overall, 4),
    }
