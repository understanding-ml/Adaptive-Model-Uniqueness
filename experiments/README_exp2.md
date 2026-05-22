# Experiment 2: Vision Ecosystems under Continuous Transformations

## Overview

This experiment studies whether adaptive search can discover image transformations
where a target model cannot be well explained by a mixture of peer models.

The experiment evaluates residual-based model substitutability under:

- continuous image transformations
- adversarial perturbations (FGSM)

---

## Goal

Given a fitted peer ensemble weight vector $\hat{w}$,
the goal is to search for transformations that maximize:

$$
R(x,\theta)=|y_t-\hat{w}^{\top}y_{peers}|
$$

where:

- $y_t$ is the target model output
- $y_{peers}$ are peer model outputs
- $R(x,\theta)$ is the residual score

Large residuals indicate regions where peer models fail to explain the target model.

---

## Experimental Setup

### Target model

- ConvNeXt

### Peer models

- ResNet
- EfficientNet
- ViT

### Transformations

Natural transformations:

- blur
- brightness
- contrast
- rotation

Adversarial perturbation:

- FGSM

---

## Search Methods

Implemented methods:

- Random Search
- CMA-ES
- Latin Hypercube Sampling (LHS)
- Grid1D

Baseline comparison:

- Pairwise disagreement

---

## Running the Experiment

Example:

```bash
python experiments/exp22_vision_adaptive.py \
    --search_method cma \
    --attack_mode none \
    --max_samples 300 \
    --n_fit_queries 150 \
    --search_budget 300 \
    --seeds 0 1 2 3 4
```

FGSM setting:

```bash
python experiments/exp22_vision_adaptive.py \
    --search_method cma \
    --attack_mode fgsm \
    --max_samples 300 \
    --n_fit_queries 150 \
    --search_budget 300 \
    --seeds 0 1 2 3 4
```

---

## Output Files

Results are stored in:

```text
results_exp22/
```

Each JSON file contains:

- configuration
- search curves
- holdout metrics
- pairwise metrics
- per-seed statistics

---

## Main Metrics

### Search metrics

- `search_best_residual_mean`
- `search_curve`

### Holdout metrics

- `holdout_mean_residual_mean`
- `holdout_max_residual_mean`

### Pairwise baseline

- `holdout_mean_pairwise_mean`

---

## Key Findings

- CMA generally discovers higher residual regions than random search.
- High residual cases mainly appear in a small number of samples (long-tail behavior).
- FGSM perturbations increase residual values compared to natural transformations.
- Pairwise disagreement and residual scores behave differently,
  suggesting that large disagreement with a single peer does not necessarily imply non-substitutability.