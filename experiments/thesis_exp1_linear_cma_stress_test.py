# experiments/exp18.py
import cma
import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Ensure the 'isqed' package can be imported from the parent directory
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


def compute_residual(target, peers, w_hat, x_vec: np.ndarray) -> float:
    """Certificate violation residual r(x) = |y_t(x) - phi_p(x)^T w_hat|."""
    X = x_vec.reshape(1, -1)
    y_t = float(target._forward(X)[0])
    phi = np.array([p._forward(X)[0] for p in peers], dtype=float)
    return abs(y_t - float(phi @ w_hat))


def stress_test_random(target, peers, w_hat, K, dim, R=3.0, seed=0):
    """
    Stress-test 阶段：固定 w_hat，随机采样 action，最大化 residual
    返回 interrogation curve：best_so_far residual vs query idx
    """
    rng = np.random.default_rng(seed)
    best = -np.inf
    curve = []

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

        r = compute_residual(target, peers, w_hat, x)
        if r > best:
            best = r

        curve.append({"query": k, "best_residual": best, "attempts": attempts})

    return pd.DataFrame(curve)


def stress_test_cma_es(target, peers, w_hat, K, dim, R=3.0, seed=0,
                       sigma0=None, popsize=None, repeat_evals=1, sigma_rob=0.05,
                       lam=0.3, robust_axis_idx=0, robust_score="penalty"):
    """
    CMA-ES for maximizing residual under feasibility ||x||<=R.

    If repeat_evals > 1, each candidate action x is evaluated on a local
    one-dimensional neighborhood along a chosen action coordinate (a dose-like axis).
    We sample scalar perturbations delta ~ N(0, sigma_rob^2), perturb only the chosen
    coordinate, project back to the feasible L2 ball, and compute residuals.

    The robust score can be one of two forms:
        robust_score = "penalty":
            score = mean(residuals) - lam * (std(residuals) / sqrt(repeat_evals))
        robust_score = "mean_only":
            score = mean(residuals)

    This is closer to the paper's idea of smoothing / penalizing local variation
    along the dose direction, rather than perturbing the full action vector.

    We count each residual evaluation as one query, so total evaluations <= K.
    We project candidates back onto the L2 ball to enforce feasibility.

    Returns:
        df_curve: DataFrame with columns [query, best_residual, attempts]
        best_x: ndarray (dim,) best found action under the robust score
    """
    rng = np.random.default_rng(seed)

    # Initial point and step size
    x0 = np.zeros(dim, dtype=float)
    if sigma0 is None:
        sigma0 = R / 2.0

    # CMA options
    opts = {
        "seed": int(seed),
        "verb_log": 0,
        "verbose": -9,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)

    es = cma.CMAEvolutionStrategy(x0, float(sigma0), opts)

    best = -np.inf
    best_x = None
    curve = []

    k = 0
    attempts = 0

    while k < K and not es.stop():
        xs = es.ask()
        fitness = []

        incomplete_batch = False

        for x in xs:
            if k >= K:
                incomplete_batch = True
                break

            attempts += 1

            # Project to feasible L2 ball
            x = np.asarray(x, dtype=float)
            norm = np.linalg.norm(x)
            if norm > R:
                x = x * (R / norm)

            residuals = []
            for rep_idx in range(repeat_evals):
                if k >= K:
                    incomplete_batch = True
                    break

                if repeat_evals == 1:
                    x_eval = x
                else:
                    x_eval = x.copy()
                    delta_scalar = float(rng.standard_normal() * float(sigma_rob))
                    x_eval[int(robust_axis_idx)] += delta_scalar
                    norm_eval = np.linalg.norm(x_eval)
                    if norm_eval > R:
                        x_eval = x_eval * (R / norm_eval)

                r = compute_residual(target, peers, w_hat, x_eval)
                residuals.append(float(r))
                k += 1

            # If we hit the query budget in the middle of repeated evaluations,
            # stop cleanly without updating CMA with a partial candidate score.
            if len(residuals) < repeat_evals:
                incomplete_batch = True
                break

            mean_r = float(np.mean(residuals))
            std_r = float(np.std(residuals))

            if robust_score == "mean_only":
                score = mean_r
            elif robust_score == "penalty":
                score = mean_r - float(lam) * (std_r / np.sqrt(repeat_evals))
            else:
                raise ValueError(f"Unknown robust_score: {robust_score}")

            if score > best:
                best = score
                best_x = x.copy()

            curve.append({"query": k, "best_residual": best, "attempts": attempts})

            # CMA-ES minimizes; we want to maximize score
            fitness.append(-score)

        # CMA-ES requires a full evaluated batch. If the batch is incomplete,
        # we already recorded all valid queries above, so stop cleanly.
        if incomplete_batch or len(fitness) < len(xs):
            break

        es.tell(xs, fitness)

    return pd.DataFrame(curve), best_x


def run_once(dim=20, n_peers=10, noise_std=0.5, margin=0.6,
             D_base=200, K=120, R=3.0, seed=0, is_unique=True, repeat_evals=1, sigma_rob=0.05,
             lam=0.3, robust_axis_idx=0, robust_score="penalty"):
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

    # 3) stress test (CMA-ES)
    df_curve, best_x = stress_test_cma_es(
        target, peers, w_hat,
        K=K, dim=dim, R=R, seed=seed,
        repeat_evals=repeat_evals,
        sigma_rob=sigma_rob,
        lam=lam,
        robust_axis_idx=robust_axis_idx,
        robust_score=robust_score
    )

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
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--Ks", type=str, default="", help="Comma-separated list of K values to run, e.g. 60,120,200. If set, overrides --K")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated list of seeds to run, e.g. 0,1,2,3,4. If set, overrides --seed")
    parser.add_argument("--repeat_evals", type=int, default=1, help="Number of repeated residual evaluations per candidate action for robust CMA objective")
    parser.add_argument("--sigma_rob", type=float, default=0.05, help="Local neighborhood perturbation scale for robust CMA when repeat_evals > 1")
    parser.add_argument("--lam", type=float, default=0.3, help="Penalty strength for local variation in the robust objective")
    parser.add_argument("--robust_axis_idx", type=int, default=0, help="Which action coordinate to perturb for route-B robust interrogation")
    parser.add_argument("--robust_score", type=str, default="penalty", choices=["penalty", "mean_only"], help="Robust scoring rule: penalty = mean - lam * stderr, mean_only = neighborhood average only")
    args = parser.parse_args()

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULT_DIR = os.path.join(ROOT_DIR, "results", "thesis_exp1")
    os.makedirs(RESULT_DIR, exist_ok=True)

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

    if args.repeat_evals == 1:
        mode_tag = ""
    else:
        mode_tag = (
            f"_rep{args.repeat_evals}_axis{args.robust_axis_idx}"
            f"_srob{str(args.sigma_rob).replace('.', 'p')}"
        )
        if args.robust_score == "penalty":
            mode_tag += f"_lam{str(args.lam).replace('.', 'p')}"
        else:
            mode_tag += "_meanonly"

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
                    is_unique=args.unique,
                    repeat_evals=args.repeat_evals,
                    sigma_rob=args.sigma_rob,
                    lam=args.lam,
                    robust_axis_idx=args.robust_axis_idx,
                    robust_score=args.robust_score
                )

                out = os.path.join(
                    RESULT_DIR,
                    f"exp18_stress_cma_curve{mode_tag}_dim{dim}_K{K}_R{args.R}_seed{seed}_unique{int(args.unique)}.csv"
                )
                df_curve.to_csv(out, index=False)

                best_x_path = os.path.join(
                    RESULT_DIR,
                    f"exp18_best_x{mode_tag}_dim{dim}_K{K}_R{args.R}_seed{seed}_unique{int(args.unique)}.npy"
                )
                if best_x is not None:
                    np.save(best_x_path, best_x)
                    print("  Saved best_x:", best_x_path)

                plot_path = os.path.join(
                    RESULT_DIR,
                    f"exp18_stress_cma_curve{mode_tag}_dim{dim}_K{K}_R{args.R}_seed{seed}_unique{int(args.unique)}.png"
                )
                plt.figure()
                plt.plot(df_curve["query"], df_curve["best_residual"])
                plt.xlabel("Query index")
                plt.ylabel("Best-so-far residual")
                title_suffix = "" if args.repeat_evals == 1 else (
                    f" (robust-axis, m={args.repeat_evals}, sigma={args.sigma_rob}, "
                    f"score={args.robust_score}, lam={args.lam}, axis={args.robust_axis_idx})"
                )
                plt.title(f"CMA-ES Stress-Test Interrogation Curve{title_suffix}")
                plt.savefig(plot_path)
                plt.close()

                max_residual = float(df_curve["best_residual"].iloc[-1])
                total_attempts = int(df_curve["attempts"].iloc[-1])

                summary_rows.append({
                    "unique": int(args.unique),
                    "dim": dim,
                    "repeat_evals": int(args.repeat_evals),
                    "sigma_rob": float(args.sigma_rob),
                    "lam": float(args.lam),
                    "robust_axis_idx": int(args.robust_axis_idx),
                    "robust_score": args.robust_score,
                    "K": K,
                    "R": float(args.R),
                    "seed": seed,
                    "baseline_disco_distance": float(dist),
                    "max_residual": max_residual,
                    "total_attempts": total_attempts,
                    "best_x_path": best_x_path,
                })

                print(
                    f"[unique={int(args.unique)} dim={dim}] K={K} seed={seed} "
                    f"-> max_residual={max_residual:.4f}, attempts={total_attempts}, dist={dist:.4f}"
                )
                print("  Saved:", out)
                print("  Saved plot:", plot_path)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        RESULT_DIR,
        f"exp18_summary{mode_tag}_R{args.R}_unique{int(args.unique)}.csv"
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
            )
        )

        stats_path = os.path.join(
            RESULT_DIR,
            f"exp18_mean_std{mode_tag}_R{args.R}_unique{int(args.unique)}.csv"
        )
        stats_df.to_csv(stats_path, index=False)
        print("Saved mean±std stats:", stats_path)

        plt.figure()
        plt.errorbar(
            stats_df["K"],
            stats_df["mean_max_residual"],
            yerr=stats_df["std_max_residual"],
            fmt="o-",
            capsize=4
        )
        plt.xlabel("Query budget K")
        plt.ylabel("Max residual (mean ± std)")
        title_suffix = "" if args.repeat_evals == 1 else (
            f" (robust-axis, m={args.repeat_evals}, sigma={args.sigma_rob}, "
            f"score={args.robust_score}, lam={args.lam}, axis={args.robust_axis_idx})"
        )
        plt.title(f"Stress Test Performance (mean ± std across seeds){title_suffix}")

        plot_stats_path = os.path.join(
            RESULT_DIR,
            f"exp18_mean_std{mode_tag}_R{args.R}_unique{int(args.unique)}.png"
        )
        plt.savefig(plot_stats_path)
        plt.close()

        print("Saved mean±std plot:", plot_stats_path)

if __name__ == "__main__":
    main()