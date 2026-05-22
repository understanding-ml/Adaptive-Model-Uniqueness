# experiments/exp22_vision_adaptive.py
#
# Exp 22: Vision adaptive stress-testing with continuous transformations.
#
# Supports:
#   1) random search
#   2) CMA-ES
#   3) robust reward mode (neighborhood-averaged residual)
#
# Pipeline:
#   1) Load ImageNet-compatible validation images via ImageFolder.
#   2) Load a standard 4-model ecosystem.
#   3) Fit a convex routing certificate w_hat on random transformed queries.
#   4) Run a chosen search strategy to find high-residual contexts.
#   5) Optionally use robust reward to reduce spike hunting.
#   6) Evaluate the discovered transformation on holdout data.

import json
import os
import sys
import time
from typing import List, Tuple, Dict

import cma
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms, models
import torch.nn.functional as F

# Make local package importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from isqed.geometry import DISCOSolver
from isqed.real_world import ImageModelWrapper

SCALAR_MODE = "p_true"

IMAGENETTE_TO_IMAGENET1K_IDX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

NATURAL_THETA_LOWER = np.array([0.0, -0.3, -0.3, -20.0], dtype=float)
NATURAL_THETA_UPPER = np.array([2.0, 0.3, 0.3, 20.0], dtype=float)
DEFAULT_FGSM_EPS_MAX = 8.0 / 255.0


def denorm(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return x * std + mean


def renorm(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def clip_theta(
        theta: np.ndarray,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
) -> np.ndarray:
    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    return np.clip(theta, lower, upper)



def get_theta_bounds(
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    if attack_mode == "fgsm":
        lower = np.concatenate([NATURAL_THETA_LOWER, np.array([0.0], dtype=float)])
        upper = np.concatenate([NATURAL_THETA_UPPER, np.array([fgsm_eps_max], dtype=float)])
        return lower, upper
    else:
        return NATURAL_THETA_LOWER.copy(), NATURAL_THETA_UPPER.copy()


# === Helper functions for unit box parameterization ===
def theta_to_unit(
        theta: np.ndarray,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
) -> np.ndarray:
    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    scale = upper - lower
    scale = np.where(scale == 0.0, 1.0, scale)
    return np.clip((np.asarray(theta, dtype=float) - lower) / scale, 0.0, 1.0)


def unit_to_theta(
        z: np.ndarray,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
) -> np.ndarray:
    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    return lower + z * (upper - lower)


class ImageContinuousTransformIntervention:
    """
    Continuous image intervention with 4 coordinates:
      theta = (blur_sigma, brightness_delta, contrast_delta, rotation_deg)
    """

    def apply(
            self,
            sample: Tuple[torch.Tensor, int],
            theta: Tuple[float, float, float, float],
            seed: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        x, y = sample
        blur_sigma, bright_delta, contrast_delta, rot_deg = theta

        x0 = denorm(x).clamp(0.0, 1.0)
        x1 = TF.adjust_brightness(x0, 1.0 + float(bright_delta))
        x1 = TF.adjust_contrast(x1, 1.0 + float(contrast_delta))
        x1 = TF.rotate(x1, angle=float(rot_deg))

        if float(blur_sigma) > 1e-8:
            x1 = TF.gaussian_blur(x1, kernel_size=5, sigma=float(blur_sigma))

        x1 = x1.clamp(0.0, 1.0)
        x1 = renorm(x1)
        return x1, y


def fgsm_attack_on_target(
        target_wrapper: ImageModelWrapper,
        sample: Tuple[torch.Tensor, int],
        epsilon: float,
) -> Tuple[torch.Tensor, int]:
    x_norm, y = sample

    device = target_wrapper.device

    # 先把输入搬到和模型同一个 device
    x_norm = x_norm.to(device)

    # 转成 pixel space
    x_pix = denorm(x_norm).clamp(0.0, 1.0).detach()

    # batch 维
    x_adv = x_pix.unsqueeze(0).clone().detach().requires_grad_(True)
    label = torch.tensor([y], dtype=torch.long, device=device)

    # 重新归一化后送入模型
    x_adv_norm = renorm(x_adv)

    logits = target_wrapper.model(x_adv_norm)
    loss = F.cross_entropy(logits, label)
    grad = torch.autograd.grad(loss, x_adv)[0]

    # FGSM
    x_adv = x_adv + float(epsilon) * grad.sign()

    # 投影回 epsilon-ball and valid pixel range
    x_low = (x_pix.unsqueeze(0) - float(epsilon)).clamp(0.0, 1.0)
    x_high = (x_pix.unsqueeze(0) + float(epsilon)).clamp(0.0, 1.0)
    x_adv = torch.max(torch.min(x_adv, x_high), x_low)
    x_adv = x_adv.clamp(0.0, 1.0).detach()

    x_adv_norm = renorm(x_adv).squeeze(0)

    # 返回时搬回 CPU，这样后面 wrapper._forward(sample) 那套接口更稳
    return x_adv_norm.cpu(), y


# ============================================================
# 0. Preprocessing
# ============================================================

def build_imagenet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ============================================================
# 1. Model loading utilities
# ============================================================

def load_standard_models(device: str) -> Dict[str, ImageModelWrapper]:
    model_wrappers: Dict[str, ImageModelWrapper] = {}

    res50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model_wrappers["ResNet50"] = ImageModelWrapper(res50, "ResNet50", device, mode=SCALAR_MODE)

    eff_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model_wrappers["EfficientNetB0"] = ImageModelWrapper(
        eff_b0, "EfficientNetB0", device, mode=SCALAR_MODE
    )

    convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model_wrappers["ConvNeXtTiny"] = ImageModelWrapper(
        convnext_tiny, "ConvNeXtTiny", device, mode=SCALAR_MODE
    )

    vit_b16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model_wrappers["ViT_B16"] = ImageModelWrapper(vit_b16, "ViT_B16", device, mode=SCALAR_MODE)

    return model_wrappers


# ============================================================
# 2. Dataset utilities
# ============================================================

def load_context_samples(
        root: str,
        max_samples: int,
        np_rng: np.random.RandomState,
        context_name: str,
):
    transform = build_imagenet_transform()
    dataset = datasets.ImageFolder(root=root, transform=transform)

    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError(f"Context '{context_name}' has no images under {root}.")

    if max_samples is not None and max_samples < n_total:
        indices = np_rng.choice(n_total, size=max_samples, replace=False)
    else:
        indices = np.arange(n_total)

    samples = []
    ids = []

    for idx in indices:
        x, _local_y = dataset[idx]
        path, _ = dataset.samples[idx]

        synset = os.path.basename(os.path.dirname(path))
        if synset not in IMAGENETTE_TO_IMAGENET1K_IDX:
            raise KeyError(f"Synset {synset} not found in Imagenette mapping.")

        imagenet_y = IMAGENETTE_TO_IMAGENET1K_IDX[synset]

        samples.append((x, int(imagenet_y)))
        ids.append(os.path.relpath(path, root))

    return samples, ids


def split_fit_search_holdout_indices(
        n: int,
        np_rng: np.random.RandomState,
        fit_fraction: float = 0.4,
        search_fraction: float = 0.3,
):
    idx = np.arange(n)
    np_rng.shuffle(idx)

    n_fit = int(n * fit_fraction)
    n_search = int(n * search_fraction)

    fit_idx = idx[:n_fit]
    search_idx = idx[n_fit:n_fit + n_search]
    holdout_idx = idx[n_fit + n_search:]
    return fit_idx, search_idx, holdout_idx


# ============================================================
# 3. Stress-test helpers
# ============================================================
def apply_action(
        target_wrapper: ImageModelWrapper,
        sample: Tuple[torch.Tensor, int],
        theta,
        intervention: ImageContinuousTransformIntervention,
        attack_mode: str = "none",
):
    """
    theta:
      - natural mode: (blur, brightness, contrast, rotation)
      - fgsm mode   : (blur, brightness, contrast, rotation, epsilon)
    """
    theta_arr = np.asarray(theta, dtype=float)

    natural_theta = tuple(float(x) for x in theta_arr[:4].tolist())
    transformed = intervention.apply(sample, natural_theta)

    if attack_mode == "fgsm":
        epsilon = float(theta_arr[4])
        transformed = fgsm_attack_on_target(
            target_wrapper=target_wrapper,
            sample=transformed,
            epsilon=epsilon,
        )

    return transformed


def sample_theta(
        rng: np.random.RandomState,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    theta = rng.uniform(lower, upper)
    return tuple(float(x) for x in theta)


def sample_lhs_thetas(
        n_samples: int,
        rng: np.random.RandomState,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
) -> np.ndarray:
    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    dim = len(lower)

    result = np.zeros((n_samples, dim), dtype=float)

    for d in range(dim):
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.uniform(low=cut[:-1], high=cut[1:], size=n_samples)
        rng.shuffle(u)
        result[:, d] = lower[d] + u * (upper[d] - lower[d])

    return result


def fit_certificate(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        samples: List[Tuple[torch.Tensor, int]],
        intervention: ImageContinuousTransformIntervention,
        rng: np.random.RandomState,
        n_fit_queries: int = 100,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
) -> np.ndarray:
    target_vals = []
    peer_rows = []

    for _ in range(n_fit_queries):
        idx = int(rng.randint(len(samples)))
        sample = samples[idx]
        theta = sample_theta(
            rng,
            attack_mode=attack_mode,
            fgsm_eps_max=fgsm_eps_max,
        )
        perturbed = apply_action(
            target_wrapper=target_wrapper,
            sample=sample,
            theta=theta,
            intervention=intervention,
            attack_mode=attack_mode,
        )

        y_t = float(target_wrapper._forward(perturbed))
        y_peers = [float(p._forward(perturbed)) for p in peer_wrappers]

        target_vals.append(y_t)
        peer_rows.append(y_peers)

    y_t_fit = np.asarray(target_vals, dtype=float).reshape(-1, 1)
    y_p_fit = np.asarray(peer_rows, dtype=float)

    _, w_hat = DISCOSolver.solve_weights_and_distance(y_t_fit, y_p_fit)
    return np.asarray(w_hat, dtype=float).flatten()


def compute_residual(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        sample: Tuple[torch.Tensor, int],
        theta,
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        attack_mode: str = "none",
):
    perturbed = apply_action(
        target_wrapper=target_wrapper,
        sample=sample,
        theta=theta,
        intervention=intervention,
        attack_mode=attack_mode,
    )
    y_t = float(target_wrapper._forward(perturbed))
    y_peers = np.asarray([float(p._forward(perturbed)) for p in peer_wrappers], dtype=float)
    y_mix = float(np.dot(w_hat, y_peers))
    residual = abs(y_t - y_mix)
    return residual, y_t, y_peers


def compute_pairwise_disagreement(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        sample: Tuple[torch.Tensor, int],
        theta: Tuple[float, ...],
        intervention: ImageContinuousTransformIntervention,
        attack_mode: str = "none",
):
    perturbed = apply_action(
        target_wrapper=target_wrapper,
        sample=sample,
        theta=theta,
        intervention=intervention,
        attack_mode=attack_mode,
    )
    y_t = float(target_wrapper._forward(perturbed))
    y_peers = np.asarray([float(p._forward(perturbed)) for p in peer_wrappers], dtype=float)
    pairwise = float(np.max(np.abs(y_t - y_peers)))
    return pairwise, y_t, y_peers


def compute_residual_robust_reward(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        sample: Tuple[torch.Tensor, int],
        theta: Tuple[float, ...],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        rng: np.random.RandomState,
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    vals = []
    theta_arr = np.asarray(theta, dtype=float)

    for _ in range(robust_k):
        delta = rng.normal(loc=0.0, scale=robust_noise_std, size=len(theta_arr))
        theta_noisy = tuple(
            clip_theta(
                theta_arr + delta,
                attack_mode=attack_mode,
                fgsm_eps_max=fgsm_eps_max,
            ).tolist()
        )
        residual, _, _ = compute_residual(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            sample=sample,
            theta=theta_noisy,
            intervention=intervention,
            w_hat=w_hat,
            attack_mode=attack_mode,
        )
        vals.append(residual)

    return float(np.mean(vals))


def compute_utility(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        sample: Tuple[torch.Tensor, int],
        theta: Tuple[float, ...],
        intervention: ImageContinuousTransformIntervention,
        attack_mode: str = "none",
) -> float:
    perturbed = apply_action(
        target_wrapper=target_wrapper,
        sample=sample,
        theta=theta,
        intervention=intervention,
        attack_mode=attack_mode,
    )
    vals = [float(p._forward(perturbed)) for p in peer_wrappers]
    return float(np.mean(vals))


def evaluate_action(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        sample: Tuple[torch.Tensor, int],
        sample_id: str,
        theta: Tuple[float, ...],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        tau: float,
        reward_mode: str = "standard",
        rng: np.random.RandomState = None,
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
):
    utility = compute_utility(
        target_wrapper=target_wrapper,
        peer_wrappers=peer_wrappers,
        sample=sample,
        theta=theta,
        intervention=intervention,
        attack_mode=attack_mode,
    )
    if utility < tau:
        return {
            "feasible": False,
            "utility": utility,
            "score": 0.0,
            "residual": 0.0,
            "pairwise": 0.0,
            "sample_id": sample_id,
            "theta": tuple(float(x) for x in theta),
        }

    residual, y_t, y_peers = compute_residual(
        target_wrapper=target_wrapper,
        peer_wrappers=peer_wrappers,
        sample=sample,
        theta=theta,
        intervention=intervention,
        w_hat=w_hat,
        attack_mode=attack_mode,
    )

    pairwise = float(np.max(np.abs(y_t - y_peers)))

    if reward_mode == "robust":
        if rng is None:
            raise ValueError("rng must be provided when reward_mode='robust'")
        score = compute_residual_robust_reward(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            sample=sample,
            theta=theta,
            intervention=intervention,
            w_hat=w_hat,
            rng=rng,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
            fgsm_eps_max=DEFAULT_FGSM_EPS_MAX,
        )
    elif reward_mode == "pairwise":
        score = pairwise
    else:
        score = residual

    return {
        "feasible": True,
        "utility": utility,
        "score": float(score),
        "residual": float(residual),
        "pairwise": float(pairwise),
        "sample_id": sample_id,
        "theta": tuple(float(x) for x in theta),
        "y_t": float(y_t),
        "y_peers": y_peers.tolist(),
    }


def run_random_search(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        samples: List[Tuple[torch.Tensor, int]],
        ids: List[str],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        rng: np.random.RandomState,
        budget: int = 200,
        tau: float = 0.15,
        reward_mode: str = "standard",
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    best = -1.0
    best_info = None
    curve = []

    for step in range(budget):
        idx = int(rng.randint(len(samples)))
        sample = samples[idx]
        theta = sample_theta(
            rng,
            attack_mode=attack_mode,
            fgsm_eps_max=fgsm_eps_max,
        )

        info = evaluate_action(

            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            sample=sample,
            sample_id=ids[idx],
            theta=theta,
            intervention=intervention,
            w_hat=w_hat,
            tau=tau,
            reward_mode=reward_mode,
            rng=rng,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
        )

        if info["feasible"] and info["score"] > best:
            best = info["score"]
            best_info = {"step": step, **info}

        curve.append(best if best >= 0 else 0.0)

    return curve, best_info


def run_1d_grid_search(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        samples: List[Tuple[torch.Tensor, int]],
        ids: List[str],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        rng: np.random.RandomState,
        budget: int = 200,
        tau: float = 0.15,
        reward_mode: str = "standard",
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    """
    One-dimensional dose grid:
    vary one coordinate at a time, keep others at 0.
    Total number of evaluated actions is approximately budget.
    """
    best = -1.0
    best_info = None
    curve = []

    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    dim = len(lower)
    points_per_dim = max(1, budget // dim)

    grids = []
    for d in range(dim):
        vals = np.linspace(lower[d], upper[d], points_per_dim)
        for v in vals:
            theta = np.zeros(dim, dtype=float)
            theta[d] = v
            grids.append(tuple(theta.tolist()))

    # 如果因为整除问题少了/多了，就裁一下
    grids = grids[:budget]

    for step, theta in enumerate(grids):
        idx = int(rng.randint(len(samples)))
        sample = samples[idx]

        info = evaluate_action(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            sample=sample,
            sample_id=ids[idx],
            theta=theta,
            intervention=intervention,
            w_hat=w_hat,
            tau=tau,
            reward_mode=reward_mode,
            rng=rng,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
        )

        if info["feasible"] and info["score"] > best:
            best = info["score"]
            best_info = {"step": step, **info}

        curve.append(best if best >= 0 else 0.0)

    return curve, best_info


def run_lhs_search(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        samples: List[Tuple[torch.Tensor, int]],
        ids: List[str],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        rng: np.random.RandomState,
        budget: int = 200,
        tau: float = 0.15,
        reward_mode: str = "standard",
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    """
    Multi-dimensional Latin hypercube baseline.
    """
    best = -1.0
    best_info = None
    curve = []

    thetas = sample_lhs_thetas(
        budget,
        rng,
        attack_mode=attack_mode,
        fgsm_eps_max=fgsm_eps_max,
    )

    for step in range(budget):
        idx = int(rng.randint(len(samples)))
        sample = samples[idx]
        theta = tuple(thetas[step].tolist())

        info = evaluate_action(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            sample=sample,
            sample_id=ids[idx],
            theta=theta,
            intervention=intervention,
            w_hat=w_hat,
            tau=tau,
            reward_mode=reward_mode,
            rng=rng,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
        )

        if info["feasible"] and info["score"] > best:
            best = info["score"]
            best_info = {"step": step, **info}

        curve.append(best if best >= 0 else 0.0)

    return curve, best_info


def run_cma_search(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        samples: List[Tuple[torch.Tensor, int]],
        ids: List[str],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        rng: np.random.RandomState,
        budget: int = 200,
        tau: float = 0.15,
        sigma0: float = 0.25,
        reward_mode: str = "standard",
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    lower, upper = get_theta_bounds(attack_mode, fgsm_eps_max)
    dim = len(lower)
    init_z = [0.5] * dim

    es = cma.CMAEvolutionStrategy(
        init_z,
        sigma0,
        {
            "bounds": [[0.0] * dim, [1.0] * dim],
            "seed": int(rng.randint(10 ** 6)),
            "verb_log": 0,
            "verbose": -9,
        },
    )

    best = -1.0
    best_info = None
    curve = []
    steps_used = 0

    while steps_used < budget:
        solutions = es.ask()
        scores = []

        for theta_vec in solutions:
            if steps_used >= budget:
                break

            idx = int(rng.randint(len(samples)))
            sample = samples[idx]
            z = np.clip(np.asarray(theta_vec, dtype=float), 0.0, 1.0)
            theta = tuple(
                unit_to_theta(
                    z,
                    attack_mode=attack_mode,
                    fgsm_eps_max=fgsm_eps_max,
                ).tolist()
            )

            info = evaluate_action(
                target_wrapper=target_wrapper,
                peer_wrappers=peer_wrappers,
                sample=sample,
                sample_id=ids[idx],
                theta=theta,
                intervention=intervention,
                w_hat=w_hat,
                tau=tau,
                reward_mode=reward_mode,
                rng=rng,
                robust_k=robust_k,
                robust_noise_std=robust_noise_std,
                attack_mode=attack_mode,
            )

            value = info["score"] if info["feasible"] else 0.0
            scores.append(-value)  # CMA-ES minimizes

            if info["feasible"] and info["score"] > best:
                best = info["score"]
                best_info = {"step": steps_used, **info}

            curve.append(best if best >= 0 else 0.0)
            steps_used += 1

        if len(scores) > 0:
            es.tell(solutions[:len(scores)], scores)

    return curve, best_info


# ============================================================
# holdout
# ============================================================

def run_holdout_evaluation(
        target_wrapper: ImageModelWrapper,
        peer_wrappers: List[ImageModelWrapper],
        samples: List[Tuple[torch.Tensor, int]],
        ids: List[str],
        intervention: ImageContinuousTransformIntervention,
        w_hat: np.ndarray,
        theta: Tuple[float, ...],
        tau: float = 0.15,
        attack_mode: str = "none",
):
    residuals = []
    utilities = []
    feasible_ids = []
    pairwises = []

    per_sample_residuals = []
    per_sample_utilities = []
    per_sample_feasible = []

    for sample, sample_id in zip(samples, ids):
        info = evaluate_action(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            sample=sample,
            sample_id=sample_id,
            theta=theta,
            intervention=intervention,
            w_hat=w_hat,
            tau=tau,
            attack_mode=attack_mode,
        )

        utility = float(info["utility"])
        feasible = bool(info["feasible"])
        residual = float(info["residual"]) if feasible else 0.0
        pairwise = float(info["pairwise"]) if feasible else 0.0

        utilities.append(utility)
        per_sample_utilities.append(utility)
        per_sample_feasible.append(feasible)
        per_sample_residuals.append(
            {
                "sample_id": sample_id,
                "feasible": feasible,
                "residual": residual,
                "pairwise": pairwise,
                "utility": utility,
            }
        )

        if feasible:
            residuals.append(residual)
            pairwises.append(pairwise)
            feasible_ids.append(sample_id)

    n_total = len(samples)
    n_feasible = len(residuals)

    summary = {
        "theta": tuple(float(x) for x in theta),
        "n_total": n_total,
        "n_feasible": n_feasible,
        "feasible_fraction": (n_feasible / n_total) if n_total > 0 else 0.0,
        "mean_utility": float(np.mean(utilities)) if len(utilities) > 0 else 0.0,
        "mean_residual": float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
        "max_residual": float(np.max(residuals)) if len(residuals) > 0 else 0.0,
        "mean_pairwise": float(np.mean(pairwises)) if len(pairwises) > 0 else 0.0,
        "max_pairwise": float(np.max(pairwises)) if len(pairwises) > 0 else 0.0,
        "feasible_ids": feasible_ids,
        "per_sample_residuals": per_sample_residuals,
    }
    return summary


# ============================================================
# 4. Main experiment
# ============================================================

def run_vision_adaptive_experiment(
        data_root: str,
        max_samples: int = 60,
        fit_fraction: float = 0.4,
        search_fraction: float = 0.3,
        n_fit_queries: int = 30,
        search_budget: int = 30,
        utility_tau: float = 0.15,
        search_method: str = "random",
        seed: int = 0,
        reward_mode: str = "standard",
        robust_k: int = 5,
        robust_noise_std: float = 0.05,
        attack_mode: str = "none",
        fgsm_eps_max: float = DEFAULT_FGSM_EPS_MAX,
):
    print("=== Exp 22: Vision adaptive stress-testing ===")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading data from: {data_root}")

    samples, ids = load_context_samples(
        root=data_root,
        max_samples=max_samples,
        np_rng=rng,
        context_name="imagenette_val",
    )
    print(f"Loaded {len(samples)} samples.")

    fit_idx, search_idx, holdout_idx = split_fit_search_holdout_indices(
        len(samples),
        np_rng=rng,
        fit_fraction=fit_fraction,
        search_fraction=search_fraction,
    )

    fit_samples = [samples[i] for i in fit_idx]
    search_samples = [samples[i] for i in search_idx]
    search_ids = [ids[i] for i in search_idx]
    holdout_samples = [samples[i] for i in holdout_idx]
    holdout_ids = [ids[i] for i in holdout_idx]

    print(
        f"Split sizes -> fit: {len(fit_samples)}, "
        f"search: {len(search_samples)}, holdout: {len(holdout_samples)}"
    )

    print("\nLoading standard models...")
    models_std = load_standard_models(device=device)
    print(f"Models loaded: {list(models_std.keys())}")

    target_name = "ConvNeXtTiny"
    peer_names = ["ResNet50", "EfficientNetB0", "ViT_B16"]
    target_wrapper = models_std[target_name]
    peer_wrappers = [models_std[name] for name in peer_names]
    print(f"Target: {target_name}")
    print(f"Peers: {peer_names}")
    print(f"Reward mode: {reward_mode}")
    print(f"Attack mode: {attack_mode}")
    if attack_mode == "fgsm":
        print(f"FGSM epsilon max: {fgsm_eps_max}")
    if reward_mode == "robust":
        print(f"Robust reward settings -> K={robust_k}, noise_std={robust_noise_std}")

    intervention = ImageContinuousTransformIntervention()

    print("\nRunning sanity check...")
    sample0 = samples[0]
    theta0 = sample_theta(
        rng,
        attack_mode=attack_mode,
        fgsm_eps_max=fgsm_eps_max,
    )
    clean_score = float(target_wrapper._forward(sample0))
    perturbed_score = float(
        target_wrapper._forward(
            apply_action(
                target_wrapper=target_wrapper,
                sample=sample0,
                theta=theta0,
                intervention=intervention,
                attack_mode=attack_mode,
            )
        )
    )
    print(f"  clean_score     = {clean_score:.6f}")
    print(f"  perturbed_score = {perturbed_score:.6f}")
    print(f"  theta0          = {theta0}")

    print("\nFitting convex routing certificate on natural transformations...")
    w_hat = fit_certificate(
        target_wrapper=target_wrapper,
        peer_wrappers=peer_wrappers,
        samples=fit_samples,
        intervention=intervention,
        rng=rng,
        n_fit_queries=n_fit_queries,
        attack_mode="none",
        fgsm_eps_max=fgsm_eps_max,
    )
    print(f"w_hat = {w_hat}")

    if search_method == "random":
        print("\nRunning constrained random search...")
        curve, best_info = run_random_search(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            samples=search_samples,
            ids=search_ids,
            intervention=intervention,
            w_hat=w_hat,
            rng=rng,
            budget=search_budget,
            tau=utility_tau,
            reward_mode=reward_mode,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
            fgsm_eps_max=fgsm_eps_max,
        )

    elif search_method == "lhs":
        print("\nRunning constrained Latin hypercube search...")
        curve, best_info = run_lhs_search(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            samples=search_samples,
            ids=search_ids,
            intervention=intervention,
            w_hat=w_hat,
            rng=rng,
            budget=search_budget,
            tau=utility_tau,
            reward_mode=reward_mode,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
            fgsm_eps_max=fgsm_eps_max,
        )

    elif search_method == "grid1d":
        print("\nRunning constrained 1D dose grid search...")
        curve, best_info = run_1d_grid_search(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            samples=search_samples,
            ids=search_ids,
            intervention=intervention,
            w_hat=w_hat,
            rng=rng,
            budget=search_budget,
            tau=utility_tau,
            reward_mode=reward_mode,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
            fgsm_eps_max=fgsm_eps_max,
        )

    elif search_method == "cma":
        print("\nRunning constrained CMA-ES search...")
        print("CMA parameterization: normalized unit box [0, 1]^d mapped to theta bounds")
        curve, best_info = run_cma_search(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            samples=search_samples,
            ids=search_ids,
            intervention=intervention,
            w_hat=w_hat,
            rng=rng,
            budget=search_budget,
            tau=utility_tau,
            reward_mode=reward_mode,
            robust_k=robust_k,
            robust_noise_std=robust_noise_std,
            attack_mode=attack_mode,
            fgsm_eps_max=fgsm_eps_max,
        )

    else:
        raise ValueError(f"Unknown search_method: {search_method}")

    print("\n=== Search finished ===")
    print(f"Search method: {search_method}")
    print(f"Reward mode: {reward_mode}")
    print(f"Best-so-far final score: {curve[-1] if len(curve) > 0 else 'N/A'}")
    print(f"Best info: {best_info}")

    holdout_summary = None
    holdout_summary_normal = None
    holdout_summary_fgsm = None

    if best_info is not None:
        print("\nRunning holdout evaluation under natural transformations...")
        holdout_summary_normal = run_holdout_evaluation(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            samples=holdout_samples,
            ids=holdout_ids,
            intervention=intervention,
            w_hat=w_hat,
            theta=best_info["theta"],
            tau=utility_tau,
            attack_mode="none",
        )

        print("\nRunning holdout evaluation under FGSM perturbations...")
        fgsm_theta = tuple(list(best_info["theta"][:4]) + [float(fgsm_eps_max)])
        holdout_summary_fgsm = run_holdout_evaluation(
            target_wrapper=target_wrapper,
            peer_wrappers=peer_wrappers,
            samples=holdout_samples,
            ids=holdout_ids,
            intervention=intervention,
            w_hat=w_hat,
            theta=fgsm_theta,
            tau=utility_tau,
            attack_mode="fgsm",
        )

        holdout_summary = holdout_summary_fgsm if attack_mode == "fgsm" else holdout_summary_normal

        print("\n=== Holdout finished ===")
        print(f"Holdout theta normal: {holdout_summary_normal['theta']}")
        print(f"Holdout normal feasible fraction: {holdout_summary_normal['feasible_fraction']:.4f}")
        print(f"Holdout normal mean utility: {holdout_summary_normal['mean_utility']:.6f}")
        print(f"Holdout normal mean residual: {holdout_summary_normal['mean_residual']:.6f}")
        print(f"Holdout normal max residual: {holdout_summary_normal['max_residual']:.6f}")
        print(f"Holdout theta FGSM: {holdout_summary_fgsm['theta']}")
        print(f"Holdout FGSM feasible fraction: {holdout_summary_fgsm['feasible_fraction']:.4f}")
        print(f"Holdout FGSM mean utility: {holdout_summary_fgsm['mean_utility']:.6f}")
        print(f"Holdout FGSM mean residual: {holdout_summary_fgsm['mean_residual']:.6f}")
        print(f"Holdout FGSM max residual: {holdout_summary_fgsm['max_residual']:.6f}")
    else:
        print("\nSkipping holdout because no feasible best action was found.")
        holdout_summary = {
            "theta": None,
            "mean_residual": None,
            "max_residual": None,
            "mean_pairwise": None,
            "max_pairwise": None,
            "mean_utility": None,
            "feasible_fraction": 0.0,
            "per_sample_residuals": [],
            "note": "No feasible best action was found in search stage.",
        }
        holdout_summary_normal = holdout_summary
        holdout_summary_fgsm = holdout_summary

    final_best_score = float(curve[-1]) if len(curve) > 0 else 0.0

    return {
        "seed": seed,
        "search_method": search_method,
        "reward_mode": reward_mode,
        "attack_mode": attack_mode,
        "best_score": final_best_score,
        "best_residual": final_best_score,  # backward compatibility
        "search_curve": [float(x) for x in curve],
        "best_info": best_info,
        "holdout_mean_residual": holdout_summary["mean_residual"],
        "holdout_max_residual": holdout_summary["max_residual"],
        "holdout_mean_utility": holdout_summary["mean_utility"],
        "holdout_feasible_fraction": holdout_summary["feasible_fraction"],
        "holdout_summary": holdout_summary,
        "holdout_summary_normal": holdout_summary_normal,
        "holdout_summary_fgsm": holdout_summary_fgsm,
        "holdout_mean_residual_normal": holdout_summary_normal["mean_residual"],
        "holdout_max_residual_normal": holdout_summary_normal["max_residual"],
        "holdout_mean_utility_normal": holdout_summary_normal["mean_utility"],
        "holdout_feasible_fraction_normal": holdout_summary_normal["feasible_fraction"],
        "holdout_mean_residual_fgsm": holdout_summary_fgsm["mean_residual"],
        "holdout_max_residual_fgsm": holdout_summary_fgsm["max_residual"],
        "holdout_mean_utility_fgsm": holdout_summary_fgsm["mean_utility"],
        "holdout_feasible_fraction_fgsm": holdout_summary_fgsm["feasible_fraction"],
        "best_pairwise": best_info["pairwise"] if best_info is not None else 0.0,
        "holdout_mean_pairwise": holdout_summary["mean_pairwise"],
        "holdout_max_pairwise": holdout_summary["max_pairwise"],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp 22: Vision adaptive stress-testing with continuous image transformations."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="~/data/imagenette2/val",
        help="Root of ImageFolder validation images.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=60,
        help="Maximum number of images to load.",
    )
    parser.add_argument(
        "--fit_fraction",
        type=float,
        default=0.4,
        help="Fraction of loaded samples used for fit split.",
    )
    parser.add_argument(
        "--search_fraction",
        type=float,
        default=0.3,
        help="Fraction of loaded samples used for search split.",
    )
    parser.add_argument(
        "--n_fit_queries",
        type=int,
        default=30,
        help="Number of random transformed fit queries used to fit w_hat.",
    )
    parser.add_argument(
        "--search_budget",
        type=int,
        default=30,
        help="Number of search queries.",
    )
    parser.add_argument(
        "--utility_tau",
        type=float,
        default=0.15,
        help="Utility threshold for feasible actions.",
    )
    parser.add_argument(
        "--search_method",
        type=str,
        default="random",
        choices=["random", "lhs", "grid1d", "cma"],
        help="Search strategy used in the stress-test.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="standard",
        choices=["standard", "robust", "pairwise"],
        help="Search reward: standard residual, neighborhood-averaged robust residual, or pairwise disagreement.",
    )
    parser.add_argument(
        "--robust_k",
        type=int,
        default=5,
        help="Number of neighborhood samples for robust reward.",
    )
    parser.add_argument(
        "--robust_noise_std",
        type=float,
        default=0.05,
        help="Std of Gaussian perturbation for robust reward.",
    )
    parser.add_argument(
        "--attack_mode",
        type=str,
        default="none",
        choices=["none", "fgsm"],
        help="Whether to add a white-box FGSM adversarial strength coordinate.",
    )
    parser.add_argument(
        "--fgsm_eps_max",
        type=float,
        default=DEFAULT_FGSM_EPS_MAX,
        help="Maximum FGSM epsilon in pixel space.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="List of random seeds",
    )

    args = parser.parse_args()

    all_results = []

    for seed in args.seeds:
        print(f"\n===== Running seed {seed} =====")

        result = run_vision_adaptive_experiment(
            data_root=os.path.expanduser(args.data_root),
            max_samples=args.max_samples,
            fit_fraction=args.fit_fraction,
            search_fraction=args.search_fraction,
            n_fit_queries=args.n_fit_queries,
            search_budget=args.search_budget,
            utility_tau=args.utility_tau,
            search_method=args.search_method,
            seed=seed,
            reward_mode=args.reward_mode,
            robust_k=args.robust_k,
            robust_noise_std=args.robust_noise_std,
            attack_mode=args.attack_mode,
            fgsm_eps_max=args.fgsm_eps_max,
        )

        all_results.append(result)
        print(
            f"Seed {seed} summary -> "
            f"search_best={result['best_residual']:.6f}, "
            f"holdout_mean={result['holdout_mean_residual']:.6f}, "
            f"holdout_max={result['holdout_max_residual']:.6f}, "
            f"holdout_utility={result['holdout_mean_utility']:.6f}, "
            f"holdout_feasible={result['holdout_feasible_fraction']:.4f}"
        )

    best_residuals = [r["best_residual"] for r in all_results]
    holdout_mean_residuals = [r["holdout_mean_residual"] for r in all_results]
    holdout_max_residuals = [r["holdout_max_residual"] for r in all_results]
    holdout_mean_utilities = [r["holdout_mean_utility"] for r in all_results]
    holdout_feasible_fractions = [r["holdout_feasible_fraction"] for r in all_results]
    best_pairwises = [r["best_pairwise"] for r in all_results]
    holdout_mean_pairwises = [r["holdout_mean_pairwise"] for r in all_results]
    holdout_max_pairwises = [r["holdout_max_pairwise"] for r in all_results]

    holdout_mean_residuals_normal = [r["holdout_mean_residual_normal"] for r in all_results]
    holdout_max_residuals_normal = [r["holdout_max_residual_normal"] for r in all_results]
    holdout_mean_utilities_normal = [r["holdout_mean_utility_normal"] for r in all_results]
    holdout_feasible_fractions_normal = [r["holdout_feasible_fraction_normal"] for r in all_results]
    holdout_mean_residuals_fgsm = [r["holdout_mean_residual_fgsm"] for r in all_results]
    holdout_max_residuals_fgsm = [r["holdout_max_residual_fgsm"] for r in all_results]
    holdout_mean_utilities_fgsm = [r["holdout_mean_utility_fgsm"] for r in all_results]
    holdout_feasible_fractions_fgsm = [r["holdout_feasible_fraction_fgsm"] for r in all_results]

    summary = {
        "method": args.search_method,
        "reward_mode": args.reward_mode,
        "seeds": args.seeds,
        "config": {
            "data_root": os.path.expanduser(args.data_root),
            "max_samples": args.max_samples,
            "fit_fraction": args.fit_fraction,
            "search_fraction": args.search_fraction,
            "n_fit_queries": args.n_fit_queries,
            "search_budget": args.search_budget,
            "utility_tau": args.utility_tau,
            "reward_mode": args.reward_mode,
            "robust_k": args.robust_k,
            "robust_noise_std": args.robust_noise_std,
            "attack_mode": args.attack_mode,
            "fgsm_eps_max": args.fgsm_eps_max,
            "cma_parameterization": "unit_box",
        },
        "metrics": {
            "search_best_residual_mean": float(np.mean(best_residuals)),
            "search_best_residual_std": float(np.std(best_residuals)),
            "holdout_mean_residual_mean": float(np.mean(holdout_mean_residuals)),
            "holdout_mean_residual_std": float(np.std(holdout_mean_residuals)),
            "holdout_max_residual_mean": float(np.mean(holdout_max_residuals)),
            "holdout_max_residual_std": float(np.std(holdout_max_residuals)),
            "holdout_mean_utility_mean": float(np.mean(holdout_mean_utilities)),
            "holdout_mean_utility_std": float(np.std(holdout_mean_utilities)),
            "holdout_feasible_fraction_mean": float(np.mean(holdout_feasible_fractions)),
            "holdout_feasible_fraction_std": float(np.std(holdout_feasible_fractions)),
            "search_best_pairwise_mean": float(np.mean(best_pairwises)),
            "search_best_pairwise_std": float(np.std(best_pairwises)),
            "holdout_mean_pairwise_mean": float(np.mean(holdout_mean_pairwises)),
            "holdout_mean_pairwise_std": float(np.std(holdout_mean_pairwises)),
            "holdout_max_pairwise_mean": float(np.mean(holdout_max_pairwises)),
            "holdout_max_pairwise_std": float(np.std(holdout_max_pairwises)),
            "holdout_mean_residual_normal_mean": float(np.mean(holdout_mean_residuals_normal)),
            "holdout_mean_residual_normal_std": float(np.std(holdout_mean_residuals_normal)),
            "holdout_max_residual_normal_mean": float(np.mean(holdout_max_residuals_normal)),
            "holdout_max_residual_normal_std": float(np.std(holdout_max_residuals_normal)),
            "holdout_mean_utility_normal_mean": float(np.mean(holdout_mean_utilities_normal)),
            "holdout_mean_utility_normal_std": float(np.std(holdout_mean_utilities_normal)),
            "holdout_feasible_fraction_normal_mean": float(np.mean(holdout_feasible_fractions_normal)),
            "holdout_feasible_fraction_normal_std": float(np.std(holdout_feasible_fractions_normal)),
            "holdout_mean_residual_fgsm_mean": float(np.mean(holdout_mean_residuals_fgsm)),
            "holdout_mean_residual_fgsm_std": float(np.std(holdout_mean_residuals_fgsm)),
            "holdout_max_residual_fgsm_mean": float(np.mean(holdout_max_residuals_fgsm)),
            "holdout_max_residual_fgsm_std": float(np.std(holdout_max_residuals_fgsm)),
            "holdout_mean_utility_fgsm_mean": float(np.mean(holdout_mean_utilities_fgsm)),
            "holdout_mean_utility_fgsm_std": float(np.std(holdout_mean_utilities_fgsm)),
            "holdout_feasible_fraction_fgsm_mean": float(np.mean(holdout_feasible_fractions_fgsm)),
            "holdout_feasible_fraction_fgsm_std": float(np.std(holdout_feasible_fractions_fgsm)),
        },
        "per_seed_results": all_results,
    }

    print("\n===== Summary =====")
    print(f"Method: {args.search_method}")
    print(f"Reward mode: {args.reward_mode}")
    print(f"Seeds: {args.seeds}")
    print(f"Search best residual: mean={np.mean(best_residuals):.6f}, std={np.std(best_residuals):.6f}")
    print(
        f"Holdout mean residual: mean={np.mean(holdout_mean_residuals):.6f}, std={np.std(holdout_mean_residuals):.6f}")
    print(f"Holdout max residual: mean={np.mean(holdout_max_residuals):.6f}, std={np.std(holdout_max_residuals):.6f}")
    print(f"Holdout mean utility: mean={np.mean(holdout_mean_utilities):.6f}, std={np.std(holdout_mean_utilities):.6f}")
    print(
        f"Holdout feasible fraction: mean={np.mean(holdout_feasible_fractions):.6f}, std={np.std(holdout_feasible_fractions):.6f}")
    print(f"Search best pairwise: mean={np.mean(best_pairwises):.6f}, std={np.std(best_pairwises):.6f}")
    print(
        f"Holdout mean pairwise: mean={np.mean(holdout_mean_pairwises):.6f}, std={np.std(holdout_mean_pairwises):.6f}")
    print(f"Holdout max pairwise: mean={np.mean(holdout_max_pairwises):.6f}, std={np.std(holdout_max_pairwises):.6f}")
    print(
        f"Holdout normal mean residual: mean={np.mean(holdout_mean_residuals_normal):.6f}, "
        f"std={np.std(holdout_mean_residuals_normal):.6f}"
    )
    print(
        f"Holdout FGSM mean residual: mean={np.mean(holdout_mean_residuals_fgsm):.6f}, "
        f"std={np.std(holdout_mean_residuals_fgsm):.6f}"
    )
    save_dir = os.path.join(ROOT_DIR, "results", "thesis_exp2")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_dir,
        f"{args.search_method}_{args.reward_mode}_{args.attack_mode}_samples{args.max_samples}_fit{args.n_fit_queries}_budget{args.search_budget}_{timestamp}.json",
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()