# experiments/exp20_holdout_eval.py

import os
import numpy as np
import pandas as pd
import argparse
import sys
# Ensure the local `isqed` package can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from isqed.synthetic import LinearStructuralModel
from isqed.geometry import DISCOSolver


def generate_ecosystem(dim, n_peers, margin_gamma, is_unique=True, rng=None):
    """和 exp1 一样：生成 target + peers"""
    if rng is None:
        rng = np.random.default_rng(0)
    peers = [LinearStructuralModel(dim) for _ in range(n_peers)]
    peer_betas_matrix = np.array([p.beta for p in peers]).T

    if not is_unique:
        weights = rng.dirichlet(np.ones(n_peers))
        beta_t = peer_betas_matrix @ weights
    else:
        center = np.mean(peer_betas_matrix, axis=1)
        perturb = rng.standard_normal(dim)
        perturb /= np.linalg.norm(perturb)
        beta_t = center + perturb * (margin_gamma + 1.0)

    target = LinearStructuralModel(dim, beta=beta_t)
    return target, peers


def fit_certificate_w_hat(target, peers, X_base):
    y_t = target._forward(X_base)
    Y_p = np.array([p._forward(X_base) for p in peers]).T

    dist, w_hat = DISCOSolver.solve_weights_and_distance(
        target_vec=y_t,
        peer_matrix=Y_p
    )
    return w_hat, dist


def compute_residual(x, target, peers, w_hat):
    X = x.reshape(1, -1)

    y_t = float(target._forward(X)[0])
    phi = np.array([p._forward(X)[0] for p in peers], dtype=float)

    r = abs(y_t - float(phi @ w_hat))
    return r


def build_ecosystem_from_betas(target_beta, peer_betas, noise_std):
    target = LinearStructuralModel(len(target_beta), beta=np.array(target_beta, dtype=float))
    target.noise_std = noise_std

    peers = []
    for beta in peer_betas:
        p = LinearStructuralModel(len(beta), beta=np.array(beta, dtype=float))
        p.noise_std = noise_std
        peers.append(p)

    return target, peers


def infer_search_residual_column(df):
    candidates = [
        "search_best_residual",
        "best_residual",
        "search_residual",
        "best_value",
        "residual",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_search_residual(search_csv, seed, method, suffix, mode, dim, K):
    if search_csv is None:
        return None
    if not os.path.exists(search_csv):
        raise FileNotFoundError(f"Search summary CSV not found: {search_csv}")

    df = pd.read_csv(search_csv)
    if "seed" not in df.columns:
        raise ValueError("Search summary CSV must contain a 'seed' column.")

    col = infer_search_residual_column(df)
    if col is None:
        raise ValueError(
            "Could not infer search residual column from search CSV. "
            "Expected one of: search_best_residual, best_residual, search_residual, best_value, residual"
        )

    sub = df[df["seed"] == seed].copy()

    if "method" in sub.columns:
        sub = sub[sub["method"] == method]
    if "suffix" in sub.columns:
        sub = sub[sub["suffix"].fillna("") == suffix]
    if "mode" in sub.columns:
        sub = sub[sub["mode"] == mode]
    if "dim" in sub.columns:
        sub = sub[sub["dim"] == dim]
    if "K" in sub.columns:
        sub = sub[sub["K"] == K]

    if len(sub) == 0:
        return None
    if len(sub) > 1:
        sub = sub.iloc[[0]]

    return float(sub.iloc[0][col])


def resolve_best_x_path(method, suffix, dim, K, R, seed, unique):
    """
    Resolve the saved best_x path for different search scripts.

    exp15 / exp18 use ..._dim{dim}_K{K}_...
    exp16 (BO) currently uses ..._d{dim}_K{K}_...
    """
    base = "results/tables"
    unique_tag = int(unique)

    candidates = []

    if method in {"exp15", "exp18"}:
        candidates.append(
            os.path.join(
                base,
                f"{method}_best_x{suffix}_dim{dim}_K{K}_R{R}_seed{seed}_unique{unique_tag}.npy"
            )
        )

    if method == "exp16":
        candidates.append(
            os.path.join(
                base,
                f"{method}_best_x{suffix}_d{dim}_K{K}_R{R}_seed{seed}_unique{unique_tag}.npy"
            )
        )
        candidates.append(
            os.path.join(
                base,
                f"{method}_best_x{suffix}_dim{dim}_K{K}_R{R}_seed{seed}_unique{unique_tag}.npy"
            )
        )

    # Fallback: try both naming conventions for any future method.
    candidates.append(
        os.path.join(
            base,
            f"{method}_best_x{suffix}_dim{dim}_K{K}_R{R}_seed{seed}_unique{unique_tag}.npy"
        )
    )
    candidates.append(
        os.path.join(
            base,
            f"{method}_best_x{suffix}_d{dim}_K{K}_R{R}_seed{seed}_unique{unique_tag}.npy"
        )
    )

    seen = set()
    deduped = []
    for p in candidates:
        if p not in seen:
            deduped.append(p)
            seen.add(p)

    for p in deduped:
        if os.path.exists(p):
            return p

    return deduped[0]


def run_holdout(best_x, dim, n_peers, noise_std, margin, D_base, seed, is_unique, n_trials,
                mode="same_ecosystem", eval_mode="refit"):

    base_rng = np.random.default_rng(seed)

    residuals = []

    # Build one fixed ecosystem from the search seed. This is the recommended
    # retest holdout for the thesis question: same audited system, fresh noise/base fit.
    fixed_target, fixed_peers = generate_ecosystem(
        dim, n_peers, margin, is_unique=is_unique, rng=np.random.default_rng(seed)
    )
    fixed_target_beta = np.array(fixed_target.beta, dtype=float)
    fixed_peer_betas = [np.array(p.beta, dtype=float) for p in fixed_peers]

    # Measurement holdout: fix the audited system and the certificate, and only
    # average over repeated noisy evaluations at the same best_x.
    fixed_measurement_target = None
    fixed_measurement_peers = None
    fixed_measurement_w_hat = None
    if mode == "same_ecosystem" and eval_mode == "measurement":
        fixed_measurement_target, fixed_measurement_peers = build_ecosystem_from_betas(
            fixed_target_beta, fixed_peer_betas, noise_std
        )
        X_base = base_rng.standard_normal((D_base, dim))
        fixed_measurement_w_hat, _ = fit_certificate_w_hat(
            fixed_measurement_target, fixed_measurement_peers, X_base
        )

    for i in range(n_trials):

        if mode == "same_ecosystem":
            if eval_mode == "measurement":
                target, peers = fixed_measurement_target, fixed_measurement_peers
                w_hat = fixed_measurement_w_hat
            elif eval_mode == "refit":
                target, peers = build_ecosystem_from_betas(
                    fixed_target_beta, fixed_peer_betas, noise_std
                )
                X_base = base_rng.standard_normal((D_base, dim))
                w_hat, _ = fit_certificate_w_hat(target, peers, X_base)
            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

        elif mode == "cross_ecosystem":
            target, peers = generate_ecosystem(
                dim, n_peers, margin, is_unique=is_unique,
                rng=np.random.default_rng(seed * 10000 + i + 1)
            )
            target.noise_std = noise_std
            for p in peers:
                p.noise_std = noise_std

            X_base = base_rng.standard_normal((D_base, dim))
            w_hat, _ = fit_certificate_w_hat(target, peers, X_base)
        else:
            raise ValueError(f"Unknown holdout mode: {mode}")

        r = compute_residual(best_x, target, peers, w_hat)
        residuals.append(float(r))

    residuals = np.asarray(residuals, dtype=float)
    return float(np.mean(residuals)), float(np.std(residuals)), residuals


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, required=True)  # exp15 or exp18
    parser.add_argument("--search_csv", type=str, default=None,
                        help="Optional search summary CSV for joining per-seed search best residuals and computing search-holdout gaps")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional filename suffix such as _rep3 for robust CMA files")
    parser.add_argument("--mode", type=str, default="same_ecosystem", choices=["same_ecosystem", "cross_ecosystem"],
                        help="Holdout mode: retest within the same ecosystem or transfer to newly sampled ecosystems")
    parser.add_argument("--eval_mode", type=str, default="refit", choices=["measurement", "refit"],
                        help="Evaluation mode within a fixed ecosystem: measurement fixes w_hat and averages only over forward noise; refit re-fits w_hat each trial")

    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--R", type=float, default=3.0)
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # In measurement mode, n_trials repeats noisy evaluation at fixed x/system/w_hat.
    # In refit mode, n_trials repeats fresh baseline refits for the same searched x.
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--tau", type=float, default=None,
                        help="Optional auxiliary threshold. Holdout mean/std and search-holdout gap remain the primary outputs.")

    parser.add_argument("--n_peers", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--D_base", type=int, default=200)

    args = parser.parse_args()

    seed_list = [int(x) for x in args.seeds.split(",")]

    rows = []
    trial_rows = []

    for seed in seed_list:

        path = resolve_best_x_path(
            method=args.method,
            suffix=args.suffix,
            dim=args.dim,
            K=args.K,
            R=args.R,
            seed=seed,
            unique=args.unique,
        )

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        best_x = np.load(path)

        search_best_residual = load_search_residual(
            args.search_csv,
            seed=seed,
            method=args.method,
            suffix=args.suffix,
            mode=args.mode,
            dim=args.dim,
            K=args.K,
        )

        mean_r, std_r, residuals = run_holdout(
            best_x,
            dim=args.dim,
            n_peers=args.n_peers,
            noise_std=args.noise,
            margin=args.margin,
            D_base=args.D_base,
            seed=seed,
            is_unique=args.unique,
            n_trials=args.n_trials,
            mode=args.mode,
            eval_mode=args.eval_mode,
        )

        gap_mean = (search_best_residual - mean_r) if search_best_residual is not None else None

        exceed_count = None
        exceed_rate = None
        if args.tau is not None:
            exceed_count = int(np.sum(residuals > float(args.tau)))
            exceed_rate = float(exceed_count / len(residuals))

        rows.append({
            "method": args.method,
            "suffix": args.suffix,
            "mode": args.mode,
            "eval_mode": args.eval_mode,
            "dim": args.dim,
            "K": args.K,
            "search_best_residual": search_best_residual,
            "gap_mean": gap_mean,
            "seed": seed,
            "unique": int(args.unique),
            "tau": args.tau,
            "holdout_mean": mean_r,
            "holdout_std": std_r,
            "exceed_count": exceed_count,
            "exceed_rate": exceed_rate,
            "holdout_residuals": ";".join(f"{x:.10f}" for x in residuals)
        })

        for trial_idx, r in enumerate(residuals):
            trial_rows.append({
                "method": args.method,
                "suffix": args.suffix,
                "mode": args.mode,
                "eval_mode": args.eval_mode,
                "dim": args.dim,
                "K": args.K,
                "search_best_residual": search_best_residual,
                "gap_from_search": (search_best_residual - float(r)) if search_best_residual is not None else None,
                "seed": seed,
                "unique": int(args.unique),
                "trial": trial_idx,
                "tau": args.tau,
                "residual": float(r),
                "is_exceed": (float(r) > float(args.tau)) if args.tau is not None else None
            })

        msg = f"{args.method} seed={seed} mode={args.mode} eval_mode={args.eval_mode} holdout_mean={mean_r:.4f} holdout_std={std_r:.4f}"
        if search_best_residual is not None:
            msg += f" search_best={search_best_residual:.4f} gap={gap_mean:.4f}"
        if args.tau is not None:
            msg += f" exceed={exceed_count}/{len(residuals)} rate={exceed_rate:.3f} tau={args.tau}"
        print(msg)

    df = pd.DataFrame(rows)
    trial_df = pd.DataFrame(trial_rows)

    out = f"results/thesis_exp1/holdout_{args.method}{args.suffix}_{args.mode}_{args.eval_mode}_dim{args.dim}_K{args.K}_unique{int(args.unique)}.csv"
    out_trials = f"results/thesis_exp1/holdout_trials_{args.method}{args.suffix}_{args.mode}_{args.eval_mode}_dim{args.dim}_K{args.K}_unique{int(args.unique)}.csv"

    df.to_csv(out, index=False)
    trial_df.to_csv(out_trials, index=False)

    print(f"Saved ({args.mode}, {args.eval_mode}):", out)
    print(f"Saved trial-level ({args.mode}, {args.eval_mode}):", out_trials)


if __name__ == "__main__":
    main()