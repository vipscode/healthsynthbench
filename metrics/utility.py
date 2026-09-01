"""
Utility metrics: is synthetic data actually useful for downstream ML?
Implements TSTR (Train on Synthetic, Test on Real) vs TRTR (Train on Real,
Test on Real) comparison, the standard protocol for synthetic-data utility.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


def utility_report(real_df, synth_df, target="outcome", seed=42):
    real_train, real_test = train_test_split(
        real_df, test_size=0.3, random_state=seed, stratify=real_df[target]
    )
    X_test, y_test = real_test.drop(columns=[target]), real_test[target]

    # TRTR: train on real, test on real (ceiling reference)
    clf_real = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf_real.fit(real_train.drop(columns=[target]), real_train[target])
    trtr_auc = roc_auc_score(y_test, clf_real.predict_proba(X_test)[:, 1])
    trtr_acc = accuracy_score(y_test, clf_real.predict(X_test))

    # TSTR: train on synthetic, test on held-out real
    clf_synth = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf_synth.fit(synth_df.drop(columns=[target]), synth_df[target])
    tstr_auc = roc_auc_score(y_test, clf_synth.predict_proba(X_test)[:, 1])
    tstr_acc = accuracy_score(y_test, clf_synth.predict(X_test))

    relative_utility = tstr_auc / trtr_auc if trtr_auc > 0 else 0

    return {
        "trtr_auc": round(trtr_auc, 4),
        "trtr_accuracy": round(trtr_acc, 4),
        "tstr_auc": round(tstr_auc, 4),
        "tstr_accuracy": round(tstr_acc, 4),
        "relative_utility_score": round(relative_utility, 4),  # 1.0 = matches real-data performance
    }
