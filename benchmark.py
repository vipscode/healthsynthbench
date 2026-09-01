"""
HealthSynthBench: benchmark synthetic health-data generators on
Fidelity, Utility, and Privacy (F-U-P) — reported jointly rather than
in isolation, which is the actual gap this project targets.

Usage:
    python benchmark.py
"""
import json
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from data.real_data import generate_real_dataset
from generators import copula_lite, noise_perturb, ctgan_baseline
from metrics.fidelity import fidelity_report
from metrics.utility import utility_report
from metrics.privacy import privacy_report

QUASI_IDENTIFIERS = ["age", "bmi", "systolic_bp"]

GENERATORS = {
    "gaussian_copula_lite": copula_lite.fit_and_sample,
    "noise_perturbation": noise_perturb.fit_and_sample,
    "ctgan": ctgan_baseline.fit_and_sample,
}


def run_benchmark():
    full_real = generate_real_dataset(n=5000, seed=42)
    # split: 'gen_source' is what generators see; 'holdout' simulates
    # real records NOT used in generation, needed for the MIA privacy test
    gen_source, holdout = train_test_split(full_real, test_size=0.3, random_state=1)

    results = {}
    for name, fn in GENERATORS.items():
        print(f"Running generator: {name}")
        synth = fn(gen_source, n_samples=len(gen_source))

        fidelity = fidelity_report(gen_source, synth)
        utility = utility_report(gen_source, synth)
        privacy = privacy_report(gen_source, synth, holdout, QUASI_IDENTIFIERS)

        composite = round(
            0.3 * fidelity["overall_fidelity_score"]
            + 0.35 * utility["relative_utility_score"]
            + 0.35 * privacy["overall_privacy_score"],
            4,
        )

        results[name] = {
            "fidelity": fidelity,
            "utility": utility,
            "privacy": privacy,
            "composite_fup_score": composite,
        }

    with open("reports/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_scorecard(results)
    return results


def print_scorecard(results):
    print("\n" + "=" * 72)
    print(f"{'Generator':<24}{'Fidelity':>12}{'Utility':>12}{'Privacy':>12}{'F-U-P':>12}")
    print("=" * 72)
    for name, r in results.items():
        print(
            f"{name:<24}"
            f"{r['fidelity']['overall_fidelity_score']:>12.3f}"
            f"{r['utility']['relative_utility_score']:>12.3f}"
            f"{r['privacy']['overall_privacy_score']:>12.3f}"
            f"{r['composite_fup_score']:>12.3f}"
        )
    print("=" * 72)
    print("Fidelity/Privacy: 1.0 = best. Utility: 1.0 = matches real-data model performance.\n")


if __name__ == "__main__":
    run_benchmark()
