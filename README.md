# Adaptive Strategies for Evaluating Model Uniqueness in AI Ecosystems

This repository contains the implementation and analysis code for my master thesis experiments on adaptive stress testing and model uniqueness auditing.

---

# Overview

The thesis studies whether a target model can be represented as a convex combination of peer models, and whether adaptive search strategies can efficiently discover failure regions where this substitutability assumption breaks.

The repository includes:

- synthetic linear experiments
- vision robustness experiments
- prompt-family adaptive search experiments on LLMs

---

# Repository Structure

```text
experiments/thesis/
    README_exp1.md  <------Example commands
    thesis_exp1_linear_random_stress_test.py
    thesis_exp1_linear_cma_stress_test.py
    thesis_exp1_linear_holdout_eval.py

    README_exp2.md. <------Example commands
    thesis_exp2_vision.py
    
    README_exp3.md. <------Example commands
    thesis_exp3_sst2.py
    thesis_exp3_mnli.py

notebooks/thesis/
    thesis_exp1.ipynb
    thesis_exp2_vision.ipynb
    thesis_exp3_sst2.ipynb
    thesis_exp3_mnli.ipynb
    thesis_exp4_model_stealing_contamination.ipynb

results/
    thesis_exp1/
    thesis_exp2/
    thesis_exp3_exp4/

isqed/
    geometry.py
    ecosystem.py
    synthetic.py
    real_world.py
```
# Experiments

## Experiment 1 — Linear Synthetic Stress Test

Goal:
- Evaluate adaptive search in a controlled linear setting
- Compare Random Search, CMA-ES, and Robust CMA
- Analyze residual scaling and holdout stability

Main methods:
- Random
- CMA-ES
- Robust CMA

## Experiment 2 — Vision Robustness Audit

Goal:
- Study adaptive search under image transformations
- Evaluate robustness under distribution shift
- Compare search-stage and holdout-stage behavior

Main components:
- image transformations
- holdout evaluation
- FGSM stress testing

## Experiment 3 — Prompt-family Adaptive Search

Goal:
- Study adaptive search over prompt-edit interventions
- Evaluate whether discovered prompt edits generalize within the same prompt family

Datasets:
- SST-2
- MNLI

Main methods:
- Random
- JointPairUCB

Additional robustness mechanism:
- support-set robustness check
- family-level validation
  
## Experiment 4 – Model Stealing & Contamination Detection

Goal:

- Evaluate whether Adaptive ISQED improves detection of near-duplicate models
- Compare search effectiveness under different redundancy levels (clone vs fine-tuned target)

Controlled targets:

- Perfect Redundancy (Clone)
- Parametric Divergence (Fine-tuned Variant)

Datasets:

- SST-2
- MNLI

Main methods:

- Random
- JointPairUCB

Evaluation:

- Search efficiency under matched query cost
- Family holdout residual distribution
- Maximum holdout residual comparison

Expected observation:

- Clone targets remain nearly indistinguishable from peer models
- Fine-tuned variants exhibit a small set of transferable high-residual contexts
- Adaptive search discovers these contexts more efficiently than random search

## Environment

Different experiments use different Python environments because of package version conflicts.
    requirements_exp1.txt
    requirements_exp2.txt
    requirements_exp3.txt

## Notebooks
All cleaned analysis notebooks are located under:
    notebooks/thesis/
These notebooks reproduce:
- plots
- summary tables
- holdout analysis
- prompt-family validation analysis
