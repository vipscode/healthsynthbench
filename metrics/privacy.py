"""
Privacy metrics: how much does synthetic data leak about real individuals?

1. Distance-to-Closest-Record (DCR): for each synthetic record, distance to
   its nearest real neighbor. Low distances = synthetic records are near-copies
   of real ones = leakage risk.
2. Membership Inference Attack (MIA) simulation: a simple thresholded-distance
   attacker tries to guess which real records were used to generate the
   synthetic set. Attack accuracy near 50% = safe (attacker no better than
   a coin flip); near 100% = high leakage.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score


def _scaled(real_df, synth_df):
    scaler = StandardScaler()
    real_s = scaler.fit_transform(real_df)
    synth_s = scaler.transform(synth_df)
    return real_s, synth_s


def dcr_report(real_df, synth_df):
    real_s, synth_s = _scaled(real_df, synth_df)
    nn = NearestNeighbors(n_neighbors=1).fit(real_s)
    dists, _ = nn.kneighbors(synth_s)
    dists = dists.flatten()
    return {
        "mean_dcr": round(float(np.mean(dists)), 4),
        "median_dcr": round(float(np.median(dists)), 4),
        "pct_synth_within_0.1_of_real": round(float(np.mean(dists < 0.1) * 100), 2),
    }


def membership_inference_attack(real_df, synth_df, holdout_df, seed=42):
    """
    holdout_df: real records NOT used to generate synth_df (the attacker's
    'unseen' comparison set). Attacker guesses 'member' if a real record's
    nearest synthetic neighbor is unusually close, vs holdout records.
    Attack accuracy ~0.5 -> no leakage signal. Higher -> generator memorized data.
    """
    scaler = StandardScaler()
    synth_s = scaler.fit_transform(synth_df)
    member_s = scaler.transform(real_df)
    nonmember_s = scaler.transform(holdout_df)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_s)
    member_dist, _ = nn.kneighbors(member_s)
    nonmember_dist, _ = nn.kneighbors(nonmember_s)

    dists = np.concatenate([member_dist.flatten(), nonmember_dist.flatten()])
    labels = np.concatenate([np.ones(len(member_dist)), np.zeros(len(nonmember_dist))])

    threshold = np.median(dists)
    preds = (dists < threshold).astype(int)  # closer distance -> guess "member"
    attack_acc = accuracy_score(labels, preds)

    return {
        "attack_accuracy": round(float(attack_acc), 4),
        "leakage_signal": round(float(abs(attack_acc - 0.5) * 2), 4),  # 0 = safe, 1 = total leak
    }


def k_anonymity_estimate(synth_df, quasi_identifiers, bins=5):
    """
    Approximate k-anonymity: bin quasi-identifier columns and find the
    smallest equivalence class size across the synthetic dataset.
    """
    df = synth_df[quasi_identifiers].copy()
    for col in quasi_identifiers:
        df[col] = pd.cut(df[col], bins=bins, labels=False)
    group_sizes = df.groupby(list(quasi_identifiers)).size()
    return {
        "k_min": int(group_sizes.min()),
        "k_median": float(group_sizes.median()),
        "num_equivalence_classes": int(len(group_sizes)),
    }


def privacy_report(real_df, synth_df, holdout_df, quasi_identifiers):
    dcr = dcr_report(real_df, synth_df)
    mia = membership_inference_attack(real_df, synth_df, holdout_df)
    kanon = k_anonymity_estimate(synth_df, quasi_identifiers)
    # composite: reward low leakage_signal and higher k_min
    privacy_score = round((1 - mia["leakage_signal"]) * 0.6 + min(kanon["k_min"] / 10, 1) * 0.4, 4)
    return {
        "distance_to_closest_record": dcr,
        "membership_inference_attack": mia,
        "k_anonymity_estimate": kanon,
        "overall_privacy_score": privacy_score,
    }
