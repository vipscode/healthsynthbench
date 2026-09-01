"""
Simulates a 'real' EHR-style tabular dataset for chronic disease risk.
In actual use, replace generate_real_dataset() with a loader for a real
(access-controlled) clinical dataset. This stand-in exists so the benchmark
is fully reproducible by anyone cloning the repo, with no data access barrier.
"""
import numpy as np
import pandas as pd


def generate_real_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.normal(55, 15, n).clip(18, 95)
    bmi = rng.normal(27, 5, n).clip(15, 55)
    systolic_bp = rng.normal(128, 18, n).clip(80, 220)
    glucose = rng.normal(110, 30, n).clip(60, 350)
    smoker = rng.binomial(1, 0.22, n)
    family_history = rng.binomial(1, 0.30, n)

    # correlated risk score -> binary outcome (diabetic / not), so utility
    # metrics have real signal to preserve, not just noise
    risk = (
        0.04 * (age - 55)
        + 0.08 * (bmi - 27)
        + 0.02 * (systolic_bp - 128)
        + 0.05 * (glucose - 110)
        + 0.8 * smoker
        + 0.6 * family_history
        + rng.normal(0, 2, n)
    )
    outcome = (risk > np.percentile(risk, 70)).astype(int)

    df = pd.DataFrame({
        "age": age.round(1),
        "bmi": bmi.round(1),
        "systolic_bp": systolic_bp.round(0),
        "glucose": glucose.round(0),
        "smoker": smoker,
        "family_history": family_history,
        "outcome": outcome,
    })
    return df


if __name__ == "__main__":
    df = generate_real_dataset()
    df.to_csv("/home/claude/healthsynthbench/data/real_data.csv", index=False)
    print(df.shape, df["outcome"].mean())
