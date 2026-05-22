# experiments/exp15_certificate_stress_test.py

import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Ensure the local `isqed` package can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from isqed.synthetic import LinearStructuralModel
from isqed.geometry import DISCOSolver


def generate_ecosystem(dim, n_peers, margin_gamma, is_unique=True, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    """和 exp1 一样：生成 target + peers"""
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
    """
    Baseline 阶段：用 D 个点拟合 w_hat
    这里最简单：Phi_-t(a) 就是 peers 在这些点的输出（每个点 N_peers 维）
    """
    # target outputs: (D,)
    y_t = target._forward(X_base)

    # peer outputs: (D, N_peers)
    Y_p = np.array([p._forward(X_base) for p in peers]).T

    # solve for w_hat that minimizes ||y_t - Y_p w||_2 subject to w in simplex
    dist, w_hat = DISCOSolver.solve_weights_and_distance(
        target_vec=y_t,
        peer_matrix=Y_p
    )
    return w_hat, dist


def stress_test_random(target, peers, w_hat, K, dim, R=3.0, seed=0):
    """
    Stress-test 阶段：固定 w_hat，随机采样 action，最大化 residual
    返回 interrogation curve：best_so_far residual vs query idx
    """
    rng = np.random.default_rng(seed)
    best = -np.inf
    curve = []
    best_x = None

    k = 0
    attempts = 0
    while k < K:
        attempts += 1
        x = rng.standard_normal(dim)          # action 的 x

        # Feasibility filter: only allow actions with bounded input magnitude
        if np.linalg.norm(x) > R:
            continue

        k += 1
        X = x.reshape(1, -1)                  # (1, dim)

        y_t = float(target._forward(X)[0])
        phi = np.array([p._forward(X)[0] for p in peers], dtype=float)  # (N_peers,)

        r = abs(y_t - float(phi @ w_hat))
        if r > best:
            best = r
            best_x = x.copy()

        curve.append({"query": k, "best_residual": best, "attempts": attempts})

    return pd.DataFrame(curve), best_x


def run_once(dim=20, n_peers=10, noise_std=0.5, margin=0.6,
             D_base=200, K=120, R=3.0, seed=0, is_unique=True):
    # 1) ecosystem
    eco_rng = np.random.default_rng(seed)
    target, peers = generate_ecosystem(dim, n_peers, margin, is_unique=is_unique, rng=eco_rng)
    target.noise_std = noise_std
    for p in peers:
        p.noise_std = noise_std

    # 2) baseline fit w_hat
    rng = np.random.default_rng(seed)
    X_base = rng.standard_normal((D_base, dim))
    w_hat, dist = fit_certificate_w_hat(target, peers, X_base)

    # 3) stress test (random)
    df_curve, best_x = stress_test_random(target, peers, w_hat, K=K, dim=dim, R=R, seed=seed)

    return w_hat, dist, df_curve, best_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--dims", type=str, default="", help="Comma-separated list of dimensions to run, e.g. 5,10,20,50. If set, overrides --dim")
    parser.add_argument("--n_peers", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.6)

    parser.add_argument("--D_base", type=int, default=200)
    parser.add_argument("--K", type=int, default=120)
    parser.add_argument("--R", type=float, default=3.0, help="Feasibility radius: keep only samples with ||x||_2 <= R")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unique", action="store_true")  # H1
    parser.add_argument("--Ks", type=str, default="", help="Comma-separated list of K values to run, e.g. 60,120,200. If set, overrides --K")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated list of seeds to run, e.g. 0,1,2,3,4. If set, overrides --seed")
    args = parser.parse_args()

    # Parse multi-run lists (if provided)
    if args.Ks.strip():
        K_list = [int(x.strip()) for x in args.Ks.split(",") if x.strip()]
    else:
        K_list = [args.K]

    if args.seeds.strip():
        seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seed_list = [args.seed]
    if args.dims.strip():
        dim_list = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    else:
        dim_list = [args.dim]

    RESULT_DIR = os.path.join(ROOT_DIR, "results", "thesis_exp1")
    os.makedirs(RESULT_DIR, exist_ok=True)

    summary_rows = []

    for dim in dim_list:
        for K in K_list:
            for seed in seed_list:
                w_hat, dist, df_curve, best_x = run_once(
                    dim=dim,
                    n_peers=args.n_peers,
                    noise_std=args.noise,
                    margin=args.margin,
                    D_base=args.D_base,
                    K=K,
                    R=args.R,
                    seed=seed,
                    is_unique=args.unique
                )

                out = os.path.join(
                     RESULT_DIR,
                    f"exp15_stress_curve_dim{dim}_K{K}_R{args.R}_seed{seed}_unique{int(args.unique)}.csv"
                )
                df_curve.to_csv(out, index=False)
                best_x_path = os.path.join(
                    RESULT_DIR,
                    f"exp15_best_x_dim{dim}_K{K}_R{args.R}_seed{seed}_unique{int(args.unique)}.npy"
                )
                if best_x is not None:
                    np.save(best_x_path, best_x)

                plot_path = os.path.join(
                    RESULT_DIR,
                    f"exp15_stress_curve_dim{dim}_K{K}_R{args.R}_seed{seed}_unique{int(args.unique)}.png"
                )
                plt.figure()
                plt.plot(df_curve["query"], df_curve["best_residual"])
                plt.xlabel("Query index")
                plt.ylabel("Best-so-far residual")
                plt.title(f"Random Stress-Test Interrogation Curve (dim={dim}, K={K})")
                plt.savefig(plot_path)
                plt.close()

                max_residual = float(df_curve["best_residual"].iloc[-1])
                total_attempts = int(df_curve["attempts"].iloc[-1])

                summary_rows.append({
                    "unique": int(args.unique),
                    "dim": dim,
                    "K": K,
                    "R": float(args.R),
                    "seed": seed,
                    "baseline_disco_distance": float(dist),
                    "max_residual": max_residual,
                    "total_attempts": total_attempts,
                })

                print(
                    f"[unique={int(args.unique)} dim={dim}] K={K} seed={seed} -> max_residual={max_residual:.4f}, attempts={total_attempts}, dist={dist:.4f}")
                print("  Saved:", out)
                print("  Saved plot:", plot_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        RESULT_DIR,
        f"exp15_summary_R{args.R}_unique{int(args.unique)}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print("Saved summary:", summary_path)

    if len(seed_list) > 1:
        stats_df = (
            summary_df
            .groupby(["unique", "dim", "K", "R"], as_index=False)
            .agg(
                mean_max_residual=("max_residual", "mean"),
                std_max_residual=("max_residual", "std"),
                mean_attempts=("total_attempts", "mean"),
                std_attempts=("total_attempts", "std"),
                mean_baseline_disco_distance=("baseline_disco_distance", "mean"),
                std_baseline_disco_distance=("baseline_disco_distance", "std"),
            )
        )

        stats_path = os.path.join(
            RESULT_DIR,
            f"exp15_mean_std_R{args.R}_unique{int(args.unique)}.csv"
        )
        stats_df.to_csv(stats_path, index=False)
        print("Saved mean±std stats:", stats_path)

        plt.figure()
        for dim in sorted(stats_df["dim"].unique()):
            sub = stats_df[stats_df["dim"] == dim].sort_values("K")
            plt.errorbar(
                sub["K"],
                sub["mean_max_residual"],
                yerr=sub["std_max_residual"],
                fmt='o-',
                capsize=4,
                label=f"dim={dim}"
            )
        plt.xlabel("Query budget K")
        plt.ylabel("Max residual (mean ± std)")
        plt.title("Random Stress Test Performance (mean ± std across seeds)")
        if len(sorted(stats_df["dim"].unique())) > 1:
            plt.legend()
        plot_stats_path = os.path.join(
            RESULT_DIR,
            f"exp15_mean_std_R{args.R}_unique{int(args.unique)}.png"
        )
        plt.savefig(plot_stats_path)
        plt.close()

        print("Saved mean±std plot:", plot_stats_path)


if __name__ == "__main__":
    main()