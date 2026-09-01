"""
Baseline generator B: naive additive-noise perturbation.
Adds Gaussian noise to each real record rather than modeling the distribution.
Deliberately a *weaker* baseline: expect higher privacy leakage (records stay
close to their real source) despite reasonable marginal fidelity. This
contrast is useful for showing the benchmark actually discriminates between
generator quality.
"""
import numpy as np
import pandas as pd


def fit_and_sample(real_df: pd.DataFrame, n_samples=None, seed=2, noise_scale=0.15):
    rng = np.random.default_rng(seed)
    n_samples = n_samples or len(real_df)
    idx = rng.integers(0, len(real_df), n_samples)
    base = real_df.iloc[idx].reset_index(drop=True).copy()

    for col in real_df.columns:
        if col in ["smoker", "family_history", "outcome"]:
            flip = rng.random(n_samples) < 0.03
            base[col] = np.where(flip, 1 - base[col], base[col])
        else:
            std = real_df[col].std()
            base[col] = base[col] + rng.normal(0, noise_scale * std, n_samples)

    return base
