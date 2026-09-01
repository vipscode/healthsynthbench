"""
Baseline generator C: CTGAN (Conditional Tabular GAN).
A GAN-based deep generative model (Xu et al., 2019), in contrast to the two
statistical baselines in this benchmark (gaussian-copula-lite, noise
perturbation). It learns a conditional generator/discriminator pair over a
mode-specific normalization of the continuous columns and a one-hot encoding
of the discrete columns, rather than fitting an explicit distributional form.
Uses the standalone `ctgan` package (https://github.com/sdv-dev/CTGAN).
"""
import numpy as np
import pandas as pd
from ctgan import CTGAN

DISCRETE_COLUMNS = ["smoker", "family_history", "outcome"]

# 300 epochs (the package default) is enough to converge on this demo-sized
# dataset (~3500 training rows) and runs in roughly 2-3 minutes on CPU:
# fixed preprocessing overhead (~20s, dominated by mode-specific normalization
# of the continuous columns) plus ~0.4s/epoch on this data. Not tuned for
# best-possible fidelity -- just a reasonable demo-scale default.
EPOCHS = 300


def fit_and_sample(real_df: pd.DataFrame, n_samples=None, seed=3):
    n_samples = n_samples or len(real_df)
    cols = real_df.columns.tolist()

    model = CTGAN(epochs=EPOCHS, verbose=False)
    model.set_random_state(seed)
    model.fit(real_df, discrete_columns=DISCRETE_COLUMNS)

    synth_df = model.sample(n_samples)
    synth_df = synth_df[cols]

    # CTGAN already samples discrete columns from the training categories
    # (0/1 here), but clamp defensively so downstream metrics never see
    # anything outside {0, 1} for these columns.
    for col in DISCRETE_COLUMNS:
        if col in synth_df.columns:
            synth_df[col] = np.clip(synth_df[col].round().astype(int), 0, 1)

    return synth_df
