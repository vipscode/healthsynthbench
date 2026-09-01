"""
Baseline generator A: 'Gaussian-copula-lite'.
Preserves each column's marginal distribution AND the correlation structure
between columns by sampling from a multivariate normal fit to rank-transformed
data, then mapping back through each column's empirical distribution.
This is a lightweight stand-in for tools like SDV's GaussianCopula synthesizer.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def fit_and_sample(real_df: pd.DataFrame, n_samples=None, seed=1):
    rng = np.random.default_rng(seed)
    n_samples = n_samples or len(real_df)
    cols = real_df.columns.tolist()
    data = real_df.values.astype(float)
    n, d = data.shape

    # rank-transform each column to uniform, then to standard normal
    u = np.zeros_like(data)
    for j in range(d):
        u[:, j] = (rankdata(data[:, j]) - 0.5) / n
    z = norm.ppf(u.clip(1e-4, 1 - 1e-4))

    corr = np.corrcoef(z, rowvar=False)
    samples_z = rng.multivariate_normal(mean=np.zeros(d), cov=corr, size=n_samples)
    samples_u = norm.cdf(samples_z)

    # map back through each column's empirical quantiles
    synth = np.zeros((n_samples, d))
    for j in range(d):
        sorted_vals = np.sort(data[:, j])
        idx = (samples_u[:, j] * (n - 1)).astype(int)
        synth[:, j] = sorted_vals[idx]

    synth_df = pd.DataFrame(synth, columns=cols)
    # round/clean binary columns back to {0,1}
    for col in ["smoker", "family_history", "outcome"]:
        if col in synth_df.columns:
            synth_df[col] = (synth_df[col] > 0.5).astype(int)
    return synth_df
