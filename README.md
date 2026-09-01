# HealthSynthBench

A benchmark for synthetic health-data generators, scored jointly on **Fidelity, Utility, and Privacy (F-U-P)** — the three axes that most synthetic-data tooling and papers report in isolation, not together. A generator that scores well on utility but leaks privacy, or is "private" but useless for downstream modeling, isn't actually solving the problem. This benchmark makes that tradeoff visible and comparable across methods.

## Why this exists

Health data is the canonical case where synthetic data matters: real records are sensitive, access-restricted, and slow to get approval for, but ML research and product development still need realistic data to iterate on. The tooling landscape (SDV, CTGAN, Synthea, differential-privacy generators) offers many ways to *produce* synthetic health data, but no standard, reproducible way to *score* the output on the tradeoff that actually determines whether it's usable: **is it realistic enough to be useful, without being identifiable enough to be a liability?**

## What it measures

| Axis | Metric | What it tells you |
|---|---|---|
| **Fidelity** | KS-test per column + correlation matrix drift | Does synthetic data statistically resemble the real distribution? |
| **Utility** | Train-Synthetic-Test-Real (TSTR) vs Train-Real-Test-Real (TRTR) | Does a model trained on synthetic data perform close to one trained on real data? |
| **Privacy** | Distance-to-closest-record, membership inference attack simulation, k-anonymity estimate | Can an attacker re-identify or infer real records from the synthetic set? |

Each generator gets a composite F-U-P score plus the full breakdown, written to `reports/benchmark_results.json`.

## Quickstart

```bash
pip install -r requirements.txt
python benchmark.py
```

This runs three baseline generators against a simulated chronic-disease-risk EHR dataset (see note on data below) and prints a comparative scorecard:

```
Generator                   Fidelity     Utility     Privacy       F-U-P
========================================================================
gaussian_copula_lite           0.988       0.960       0.635       0.855
noise_perturbation             0.987       1.108       0.456       0.844
ctgan                          0.909       0.923       0.625       0.815
```

Note how `noise_perturbation` — which nearly copies real records with small perturbations — scores *higher* on utility but visibly *lower* on privacy. That's the tradeoff the benchmark is designed to surface, not an error. `ctgan` is a GAN-based deep generative baseline (Xu et al., 2019), unlike the other two statistical baselines, so it's a meaningfully different comparison point rather than just a third entry.

## On the dataset

This repo ships with a **simulated** EHR-style dataset (`data/real_data.py`) rather than real clinical data, so the benchmark is runnable by anyone without a data access agreement. The harness is dataset-agnostic — swap in any real (access-controlled) tabular clinical dataset by replacing `generate_real_dataset()` with a loader, and the same metrics apply.

## Adding a new generator

Any generator that implements `fit_and_sample(real_df, n_samples) -> synth_df` can be dropped into `generators/` and registered in `benchmark.py`'s `GENERATORS` dict — e.g. SDV's TVAE, a differential-privacy generator, or your own model.

## Roadmap

- [ ] Add a differentially-private generator (DP-SGD or PATE-based) to test the privacy axis against a method with a formal guarantee
- [ ] Support multi-table / relational EHR schemas (visits, labs, meds) instead of single flat tables
- [ ] Static leaderboard page (GitHub Pages) auto-generated from `reports/`
- [ ] Attribute-inference attack as a second privacy metric alongside membership inference

## Motivation

Built as a self-directed project applying data-privacy and synthetic-data techniques from prior health-tech data engineering work to a reproducible, open benchmark — an area with real relevance to how health data ecosystems evaluate and trust synthetic data pipelines.
