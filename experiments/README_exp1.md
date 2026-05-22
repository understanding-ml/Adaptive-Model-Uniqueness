# Experiment 1: Linear Structural Stress Test

## Overview

This experiment evaluates whether adaptive search methods can efficiently discover high-residual regions in a synthetic linear ecosystem.

The experiment follows a three-stage pipeline:

1. Fitting stage
2. Search / Audit stage
3. Holdout evaluation stage

The main goal is to determine whether large search-stage residuals correspond to genuine model uniqueness or merely exploit noise.

---

## Experiment Setup

We use a local linear structural model:

$$
Y_j(x) = \phi(x)^\top \beta_j
$$

where each model has its own coefficient vector.

Two settings are considered:

- `unique=0`
  - target model lies inside the convex hull of peers
  - large residuals should mainly come from noise

- `unique=1`
  - target model lies outside the convex hull
  - persistent residuals indicate genuine uniqueness

---

## Methods

### 1. Random Search

Uniformly samples actions from the input space.

Script:

```bash
python experiments/exp15_random_stress_test.py \
  --dim 20 \
  --Ks 200,500,1000 \
  --R 3.0 \
  --noise 0.2 \
  --margin 1.5 \
  --seeds 0,1,2,3,4

For unique target experiments:
python experiments/exp15_random_stress_test.py \
  --dim 20 \
  --Ks 200,500,1000 \
  --R 3.0 \
  --noise 0.2 \
  --margin 1.5 \
  --seeds 0,1,2,3,4 \
  --unique
```

### 2. CMA-ES Search
Uses CMA-ES to adaptively search for high-residual actions.
Script:

```bash
python experiments/exp18_cma_es_stress_test.py \
  --dim 20 \
  --Ks 200,500,1000 \
  --R 3.0 \
  --noise 0.2 \
  --margin 1.5 \
  --seeds 0,1,2,3,4
```
For unique target experiments:
Script:

```bash
python experiments/exp18_cma_es_stress_test.py \
  --dim 20 \
  --Ks 200,500,1000 \
  --R 3.0 \
  --noise 0.2 \
  --margin 1.5 \
  --seeds 0,1,2,3,4 \
  --unique
```
### 3. Robust CMA-ES
Robust CMA evaluates each candidate multiple times and penalizes unstable residual spikes.
Script:
```bash
python experiments/exp18_cma_es_stress_test.py \
  --dim 20 \
  --Ks 200,500,1000 \
  --R 3.0 \
  --noise 0.2 \
  --margin 1.5 \
  --seeds 0,1,2,3,4 \
  --repeat_evals 5 \
  --sigma_rob 0.2 \
  --lam 0.3 \
  --robust_axis_idx 0 \
  --robust_score_penalty
```
For unique target experiments:
```bash
python experiments/exp18_cma_es_stress_test.py \
  --dim 20 \
  --Ks 200,500,1000 \
  --R 3.0 \
  --noise 0.2 \
  --margin 1.5 \
  --seeds 0,1,2,3,4 \
  --repeat_evals 5 \
  --sigma_rob 0.2 \
  --lam 0.3 \
  --robust_axis_idx 0 \
  --robust_score_penalty \
  --unique
```
## Holdout Evaluation

After search, the best action found by each method is re-evaluated in two ways:

### Denoising evaluation

- Fix action
- Fix certificate
- Re-sample output randomness

This tests whether the residual is stable.

### Refit evaluation

- Re-generate baseline dataset
- Refit the certificate
- Re-evaluate residual

This tests generalization under distribution shift.

---

## Main Findings

- Adaptive search finds larger residuals than random search.
- Search-stage residual alone is insufficient.
- Some large residuals disappear during holdout evaluation.
- Robust CMA reduces overfitting to noisy residual spikes.
- Persistent holdout residuals provide stronger evidence of genuine uniqueness.