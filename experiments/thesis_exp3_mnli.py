# experiments/exp23_prompt_bandit_mnli.py
#
# In-silico ecosystem audit on MNLI with a DISCO-style dose-splitting design:
# - P_fit: low-dose prompt edits used to learn convex peer weights
# - P_eval: higher-intensity prompt edits used to evaluate PIER
#
# For each target model, we:
#   1) Collect a batch of (text, low-dose) responses from target and peers
#   2) Solve a single convex projection problem to obtain a global weight vector w_hat
#   3) Use this fixed w_hat to compute PIER on an independent batch of prompt edits
#      sampled by finite-budget interrogators; the primary search metric is
#      best-so-far max PIER, matching Exp3's rare prompt-regime stress test.

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch
import argparse
import hashlib
import re

# Ensure we can import the local `isqed` package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isqed.real_world import HuggingFaceWrapper
from isqed.geometry import DISCOSolver
from isqed.ecosystem import Ecosystem

# Import stable seed helper from experiments/utils.py
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)
from experiments.utils import make_stable_seed

# Discrete prompt-edit action ids. 0 means no edit; 1--39 are prompt-edit operators.
# P_fit uses only weak/low-dose prompt edits to fit the convex certificate.
# P_eval/search uses the full prompt-edit set to stress-test the certificate.
DOSES_FIT = np.array([0, 1, 2, 3], dtype=float)
DOSES_EVAL = np.arange(40, dtype=float)
SEARCH_BUDGET = 40
UCB_C = 1.0
QUANTILE_UCB_C = 1.0
QUANTILE_UCB_Q = 0.90
CEM_ELITE_FRAC = 0.15
CEM_SMOOTHING = 0.75
CEM_MIN_PROB = 0.005
TOP_K_HOLDOUT = 3


def prompt_family(theta):
    """Map MNLI prompt-edit action id to a coarse prompt family."""
    theta = int(round(float(theta)))
    if theta == 0:
        return "baseline"
    if 1 <= theta <= 18:
        return "style_careful"
    if 19 <= theta <= 29:
        return "attention_shift"
    if 30 <= theta <= 39:
        return "distractor_challenge"
    return "unknown"
ROBUST_SUPPORT_SIZE = 5
ROBUST_PRIMARY_WEIGHT = 0.5

class PromptEditIntervention:
    """Discrete prompt-edit intervention for MNLI-style premise/hypothesis pairs.

    The action space contains weak formatting edits, careful-reasoning prompts,
    and controlled distractor prompts. The edits preserve the original premise
    and hypothesis and do not reveal a target label.
    """

    def __init__(self):
        self.templates = [
            # 0: no prompt edit / minimal pair format
            "Premise: {premise}\nHypothesis: {hypothesis}",

            # 1--9: weak style / label-format changes
            "Premise: {premise}\nHypothesis: {hypothesis}\nRelationship:",
            "Classify the relationship between the premise and hypothesis: Premise: {premise} Hypothesis: {hypothesis}",
            "Does the premise entail, contradict, or neither entail nor contradict the hypothesis? Premise: {premise} Hypothesis: {hypothesis}",
            "Choose one label: entailment, neutral, or contradiction. Premise: {premise} Hypothesis: {hypothesis}",
            "Read the following pair and decide the NLI label. Premise: {premise} Hypothesis: {hypothesis}",
            "For MNLI natural language inference, classify this pair. Premise: {premise} Hypothesis: {hypothesis}",
            "NLI input. Premise: \"{premise}\" Hypothesis: \"{hypothesis}\" Label:",
            "A model must judge whether the hypothesis follows from the premise. Premise: {premise} Hypothesis: {hypothesis}",
            "Answer with only entailment, neutral, or contradiction: Premise: {premise} Hypothesis: {hypothesis}",

            # 10--18: careful reasoning / ambiguity framing
            "Read carefully and judge whether the premise supports, contradicts, or is unrelated to the hypothesis. Premise: {premise} Hypothesis: {hypothesis}",
            "The relationship may be subtle. Decide carefully. Premise: {premise} Hypothesis: {hypothesis}",
            "Do not rely on word overlap alone. Determine the NLI label. Premise: {premise} Hypothesis: {hypothesis}",
            "The pair may contain misleading lexical overlap. Judge the logical relationship. Premise: {premise} Hypothesis: {hypothesis}",
            "Consider whether the hypothesis must be true, must be false, or may be unknown given the premise. Premise: {premise} Hypothesis: {hypothesis}",
            "Even if the wording is similar, decide the actual logical relation. Premise: {premise} Hypothesis: {hypothesis}",
            "Separate surface similarity from logical implication. Premise: {premise} Hypothesis: {hypothesis}",
            "The pair may include negation or quantifier changes. Decide the relation. Premise: {premise} Hypothesis: {hypothesis}",
            "The pair may include a subtle contradiction or missing information. Decide the NLI label. Premise: {premise} Hypothesis: {hypothesis}",

            # 19--29: controlled distractors / attention shifts
            "Focus on whether every part of the hypothesis is supported by the premise. Premise: {premise} Hypothesis: {hypothesis}",
            "Pay special attention to negation, numbers, and quantifiers. Premise: {premise} Hypothesis: {hypothesis}",
            "Ignore irrelevant wording overlap and focus on logical consistency. Premise: {premise} Hypothesis: {hypothesis}",
            "First identify facts stated in the premise, then compare the hypothesis. Premise: {premise} Hypothesis: {hypothesis}",
            "Focus on differences between the premise and hypothesis before deciding. Premise: {premise} Hypothesis: {hypothesis}",
            "If the hypothesis adds information not guaranteed by the premise, account for that. Premise: {premise} Hypothesis: {hypothesis}",
            "If the hypothesis reverses or changes an event, account for that. Premise: {premise} Hypothesis: {hypothesis}",
            "Treat entity substitutions and role changes carefully. Premise: {premise} Hypothesis: {hypothesis}",
            "Classify after checking whether the premise rules out the hypothesis. Premise: {premise} Hypothesis: {hypothesis}",
            "Classify after checking whether the premise fully entails the hypothesis. Premise: {premise} Hypothesis: {hypothesis}",
            "Ignore stylistic similarity and focus only on inferential relation. Premise: {premise} Hypothesis: {hypothesis}",

            # 30--39: stronger but still semantics-preserving challenge prompts
            "This pair may appear easy but could contain a subtle logical mismatch. Decide the NLI label. Premise: {premise} Hypothesis: {hypothesis}",
            "Do not assume similar words imply entailment. Decide the relation. Premise: {premise} Hypothesis: {hypothesis}",
            "Do not assume different words imply contradiction. Decide the relation. Premise: {premise} Hypothesis: {hypothesis}",
            "If the premise does not provide enough evidence, choose the appropriate NLI label. Premise: {premise} Hypothesis: {hypothesis}",
            "If the hypothesis conflicts with any premise detail, account for it. Premise: {premise} Hypothesis: {hypothesis}",
            "If the hypothesis is plausible but not guaranteed, account for it. Premise: {premise} Hypothesis: {hypothesis}",
            "Classify while being cautious about misleading surface cues. Premise: {premise} Hypothesis: {hypothesis}",
            "Judge the relation from logical implication rather than shared keywords. Premise: {premise} Hypothesis: {hypothesis}",
            "First identify the key claim in the hypothesis, then compare it to the premise. Premise: {premise} Hypothesis: {hypothesis}",
            "This is a stress-test prompt. Preserve the original pair meaning and classify the NLI relation. Premise: {premise} Hypothesis: {hypothesis}",
        ]

    def _unpack(self, x):
        if isinstance(x, dict):
            return str(x.get("premise", "")), str(x.get("hypothesis", ""))
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            return str(x[0]), str(x[1])
        text = str(x)
        return text, ""

    def apply(self, x, theta, seed=None):
        premise, hypothesis = self._unpack(x)
        action = int(round(float(theta)))
        action = int(np.clip(action, 0, len(self.templates) - 1))
        return self.templates[action].format(premise=premise, hypothesis=hypothesis)

    def __call__(self, x, theta, seed=None):
        return self.apply(x, theta, seed)


HARD_SUBSET_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"n't\b",
    r"\bno\b",
    r"\bnone\b",
    r"\bwithout\b",
    r"\bexcept\b",
    r"\bonly\b",
    r"\ball\b",
    r"\bevery\b",
    r"\bsome\b",
    r"\bmost\b",
    r"\bfew\b",
    r"\bmore\b",
    r"\bless\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bwhile\b",
    r"\balthough\b",
    r"\bbut\b",
    r"\bhowever\b",
]



def pair_to_text(pair):
    """Return a stable string representation for hashing/seeding and text features."""
    if isinstance(pair, dict):
        premise = str(pair.get("premise", ""))
        hypothesis = str(pair.get("hypothesis", ""))
    elif isinstance(pair, (tuple, list)) and len(pair) >= 2:
        premise = str(pair[0])
        hypothesis = str(pair[1])
    else:
        premise = str(pair)
        hypothesis = ""
    return f"Premise: {premise}\nHypothesis: {hypothesis}"


def is_challenging_mnli_pair(pair):
    """Heuristic filter for linguistically harder MNLI examples.

    This keeps the original MNLI task unchanged, but prioritizes pairs with
    negation, quantifiers, contrast markers, length, or high lexical overlap.
    """
    if isinstance(pair, dict):
        premise = str(pair.get("premise", ""))
        hypothesis = str(pair.get("hypothesis", ""))
    else:
        premise, hypothesis = pair[0], pair[1]
        premise = str(premise)
        hypothesis = str(hypothesis)

    combined = f"{premise} {hypothesis}".lower()
    n_tokens = len(combined.split())
    if n_tokens >= 30:
        return True
    if any(re.search(pattern, combined) for pattern in HARD_SUBSET_PATTERNS):
        return True

    premise_tokens = set(re.findall(r"[A-Za-z']+", premise.lower()))
    hypothesis_tokens = set(re.findall(r"[A-Za-z']+", hypothesis.lower()))
    if len(hypothesis_tokens) > 0:
        overlap = len(premise_tokens & hypothesis_tokens) / len(hypothesis_tokens)
        if overlap >= 0.6:
            return True
    return False







# ------------------------------------------------------------------------
# Joint UCB policy over (text_idx, theta) pairs for prompt-edit bandit
# ------------------------------------------------------------------------

class JointPairUCBPolicy:
    """UCB policy over full audit actions (text_idx, theta).

    This is closer to Exp3: the interrogator chooses the whole prompt regime,
    not only the prompt edit for a preselected text.
    """

    def __init__(self, pair_actions, c=1.0, prior_scores=None):
        self.pair_actions = [(int(i), float(theta)) for i, theta in pair_actions]
        self.c = float(c)
        self.counts = {a: 0 for a in self.pair_actions}
        self.values = {a: 0.0 for a in self.pair_actions}
        self.total_steps = 0
        self.prior_scores = prior_scores or {a: 0.0 for a in self.pair_actions}

    def select(self):
        best_action = None
        best_score = -np.inf
        log_term = np.log(max(self.total_steps + 1, 2))

        for action in self.pair_actions:
            n = self.counts[action]
            prior = float(self.prior_scores.get(action, 0.0))
            if n == 0:
                # Prefer high-risk text/action pairs early, while still allowing exploration.
                score = prior + self.c
            else:
                mean = self.values[action]
                bonus = self.c * np.sqrt(log_term / n)
                score = mean + bonus

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, action, reward):
        action = (int(action[0]), float(action[1]))
        reward = float(reward)
        self.total_steps += 1
        self.counts[action] += 1
        n = self.counts[action]
        old = self.values[action]
        self.values[action] = old + (reward - old) / n







def run_bert_experiment(
        seed=0,
        max_samples=200,
        search_budget=100,
        top_k_holdout=TOP_K_HOLDOUT,
        holdout_mode="family",
        methods="all",
        run_tag="",
):
    print("--- Running Exp 23-MNLI (Prompt-Edit DISCO-style BERT Audit, via Ecosystem) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Seed: {seed}")
    experiment_seed = int(seed)
    top_k_holdout = int(top_k_holdout)
    holdout_mode = "family"

    valid_methods = {"Random", "JointPairUCB"}
    if str(methods).lower().strip() == "all":
        enabled_methods = set(valid_methods)
    else:
        enabled_methods = {m.strip() for m in str(methods).split(",") if m.strip()}
    unknown_methods = enabled_methods - valid_methods
    if unknown_methods:
        raise ValueError(
            f"Unknown method(s): {sorted(unknown_methods)}. "
            f"Valid methods: {sorted(valid_methods)}"
        )
    print(f"Enabled methods: {sorted(enabled_methods)}")

    run_tag = str(run_tag).strip()
    if run_tag:
        run_tag_suffix = f"_{run_tag}"
    else:
        method_tag = "_".join(sorted(enabled_methods)).lower()
        run_tag_suffix = f"_{method_tag}"

    # Fix PyTorch RNG for reproducibility
    torch.manual_seed(seed)

    # =====================================================================
    # 1. Define the peer ecosystem
    # =====================================================================
    peer_ids = [
        "textattack/bert-base-uncased-MNLI",
        "textattack/distilbert-base-uncased-MNLI",  # will also be used as a redundant target if available
        "textattack/roberta-base-MNLI",
        "textattack/xlnet-base-cased-MNLI",
    ]

    peers = []
    print("Loading peers...")
    for pid in peer_ids:
        try:
            model = HuggingFaceWrapper(pid, device)
            peers.append(model)
            print(f"  [OK] Loaded peer: {pid}")
        except Exception as e:
            print(f"  [SKIP] Failed to load {pid}: {e}")

    if not peers:
        print("No peers loaded. Abort.")
        return

    # =====================================================================
    # 2. Define targets
    # =====================================================================
    targets = []

    # Case A: DeBERTa MNLI target (architectural divergence)
    try:
        deberta = HuggingFaceWrapper("microsoft/deberta-base-mnli", device)
        targets.append(
            {
                "model": deberta,
                "name": "Architectural Divergence (DeBERTa-MNLI)",
                "type": "Low Redundancy",
            }
        )
    except Exception as e:
        print(f"Warning: DeBERTa MNLI target failed to load: {e}")

    # Case B: DistilBERT clone (exact peer)
    distil_ref = next((p for p in peers if "distilbert" in p.name.lower()), None)
    if distil_ref:
        targets.append(
            {
                "model": distil_ref,
                "name": "Perfect Redundancy (Clone)",
                "type": "High Redundancy",
                "clone_of_peer_idx": peers.index(distil_ref),
            }
        )
    else:
        print("Warning: No DistilBERT peer found for redundant target.")

    # Case C: DistilBERT MNLI fine-tuned variant (near-duplicate / contamination target)
    try:
        distil_mnli_variant = HuggingFaceWrapper("typeform/distilbert-base-uncased-mnli", device)
        targets.append(
            {
                "model": distil_mnli_variant,
                "name": "Parametric Divergence (DistilBERT-MNLI-Variant)",
                "type": "Uniqueness",
            }
        )
    except Exception as e:
        print(f"Warning: DistilBERT MNLI variant target failed to load: {e}")

    if not targets:
        print("No targets loaded. Abort.")
        return

    # =====================================================================
    # 3. Data and dose design (P_fit vs P_eval)
    # =====================================================================
    intervention = PromptEditIntervention()

    print("Loading MNLI validation_matched data...")
    try:
        dataset = load_dataset("glue", "mnli", split="validation_matched")
        raw_pairs = [
            {"premise": str(ex["premise"]), "hypothesis": str(ex["hypothesis"])}
            for ex in dataset
        ]
        hard_pairs = [pair for pair in raw_pairs if is_challenging_mnli_pair(pair)]
        easy_pairs = [pair for pair in raw_pairs if not is_challenging_mnli_pair(pair)]

        if len(hard_pairs) >= max_samples:
            all_sentences = hard_pairs[:max_samples]
        else:
            all_sentences = (hard_pairs + easy_pairs)[:max_samples]
        print(
            f"  [Data] Hard MNLI candidates: {len(hard_pairs)}/{len(raw_pairs)}; "
            f"using {len(all_sentences)} examples."
        )
    except Exception as e:
        print(f"  [WARN] Failed to load MNLI from HF: {e}")
        all_sentences = [
            {"premise": "A man is playing a guitar on stage.", "hypothesis": "A person is performing music."},
            {"premise": "Two children are running through a park.", "hypothesis": "No children are outdoors."},
            {"premise": "A woman is reading a book in a cafe.", "hypothesis": "A woman is sitting inside."},
            {"premise": "The dog is sleeping on the sofa.", "hypothesis": "An animal is awake outside."},
            {"premise": "Several people are waiting in line at a store.", "hypothesis": "People are shopping."},
        ] * 40

    rng = np.random.RandomState(seed)
    all_sentences = np.array(all_sentences, dtype=object)
    rng.shuffle(all_sentences)
    n_total = len(all_sentences)
    n_fit = n_total // 2
    fit_texts = all_sentences[:n_fit].tolist()
    eval_texts_all = all_sentences[n_fit:].tolist()

    n_eval_search = len(eval_texts_all) // 2
    eval_texts = eval_texts_all[:n_eval_search]
    holdout_texts = eval_texts_all[n_eval_search:]

    print(
        f"Total sentences: {n_total}, fit: {len(fit_texts)}, "
        f"search_eval: {len(eval_texts)}, holdout: {len(holdout_texts)}"
    )

    doses_fit = DOSES_FIT
    doses_eval = DOSES_EVAL
    search_budget = int(search_budget)

    print(f"P_fit doses (low):  {doses_fit}")
    print(f"Random-search action set (eval, {len(doses_eval)} actions): {doses_eval}")
    print(f"Random-search budget: {search_budget}")
    joint_ucb_c = 1.0
    robust_support_size = ROBUST_SUPPORT_SIZE
    robust_primary_weight = ROBUST_PRIMARY_WEIGHT
    print(f"JointPairUCB exploration constant: {joint_ucb_c}")
    print(f"Robust JointPairUCB support size: {robust_support_size}")
    print(f"Robust JointPairUCB primary weight: {robust_primary_weight}")

    # =====================================================================
    # 4. Main loop: DISCO-style audit per target (via Ecosystem)
    # =====================================================================
    all_results = []
    all_trajectories = []
    all_holdout_results = []
    all_holdout_pointwise_results = []

    for t_info in targets:
        t_name = t_info["name"]
        t_model = t_info["model"]
        print(f"\n>>> Auditing target: {t_name}")

        # Build an Ecosystem object where the target is audited against all peers
        eco = Ecosystem(target=t_model, peers=peers)

        # -------------------------------------------------
        # 4.1 Fit phase: learn a global convex baseline on P_fit
        # -------------------------------------------------
        print("  [Phase] Fitting convex baseline on low-dose interventions (P_fit)...")

        fit_X = []
        fit_Theta = []
        fit_seeds = []

        for text in fit_texts:
            for theta in doses_fit:
                # Deterministic intervention seed
                stable_seed = make_stable_seed(text=pair_to_text(text), theta=float(theta))
                fit_X.append(text)
                fit_Theta.append(float(theta))
                fit_seeds.append(int(stable_seed))

        if not fit_X:
            print("  [WARN] No fit samples for this target. Skipping.")
            continue

        # Query target and peers jointly via Ecosystem
        y_t_fit, Y_p_fit = eco.batched_query(
            X=fit_X,
            Thetas=fit_Theta,
            intervention=intervention,
            seeds=fit_seeds,
        )

        # Optional sanity check for clone target on P_fit
        if "clone_of_peer_idx" in t_info:
            clone_idx = t_info["clone_of_peer_idx"]
            diffs = np.abs(y_t_fit - Y_p_fit[:, clone_idx])
            max_diff = float(np.max(diffs))
            if max_diff > 1e-9:
                print(
                    f"  [ALARM] Clone mismatch on P_fit for {t_name}! "
                    f"Max diff: {max_diff:.3e}"
                )

        # Prepare data for DISCOSolver
        y_t_fit_vec = y_t_fit.reshape(-1, 1)  # (m_fit, 1)
        Y_p_fit_mat = Y_p_fit                 # (m_fit, N_peers)

        try:
            dist_fit, w_hat = DISCOSolver.solve_weights_and_distance(
                y_t_fit_vec,
                Y_p_fit_mat,
            )
        except Exception as e:
            print(f"  [ERROR] DISCOSolver failed during fit phase for {t_name}: {e}")
            continue

        w_hat = np.asarray(w_hat, dtype=float).flatten()
        if w_hat.shape[0] != len(peers):
            print(f"  [WARN] w_hat length {w_hat.shape[0]} != num_peers {len(peers)}")

        print(f"  [Fit] Learned convex weights for {t_name}: {w_hat}")

        # -------------------------------------------------
        # 4.1b Zero-dose probe on search texts for contextual features and text selection
        # -------------------------------------------------
        probe_risk_scores = None
        risk_pool_indices = None

        if len(eval_texts) > 0:
            print("  [Phase] Precomputing zero-dose risk features for contextual search...")
            probe_X = list(eval_texts)
            probe_Theta = [0.0 for _ in probe_X]
            probe_seeds = [int(make_stable_seed(text=pair_to_text(text), theta=0.0)) for text in probe_X]

            y_t_probe_all, Y_p_probe_all = eco.batched_query(
                X=probe_X,
                Thetas=probe_Theta,
                intervention=intervention,
                seeds=probe_seeds,
            )

            peer_mix_probe_all = Y_p_probe_all @ w_hat
            zero_residual_all = np.abs(y_t_probe_all - peer_mix_probe_all)
            peer_std_all = np.std(Y_p_probe_all, axis=1)
            target_conf_all = np.abs(y_t_probe_all - 0.5)

            # Higher risk means the unedited text already stresses the fitted certificate.
            # Contextual search prioritizes this pool so it is not limited by text order.
            probe_risk_scores = zero_residual_all + 0.5 * peer_std_all + 0.1 * target_conf_all
            risk_order = np.argsort(-probe_risk_scores)
            risk_pool_size = min(len(eval_texts), max(10, len(eval_texts) // 3))
            risk_pool_indices = risk_order[:risk_pool_size]
            print(
                f"  [Context] Risk-aware text pool: {risk_pool_size}/{len(eval_texts)} "
                f"texts selected for contextual search."
            )

        # Baseline-subtracted reward for adaptive selection.
        # Raw PIER still measures certificate violation, but the policy reward
        # removes the zero-dose residual of the same text so adaptive policies
        # learn prompt-edit lift rather than merely chasing naturally hard pairs.
        if len(eval_texts) > 0 and probe_risk_scores is not None:
            zero_residual_by_text = np.asarray(zero_residual_all, dtype=float)
        else:
            zero_residual_by_text = np.zeros(len(eval_texts), dtype=float)

        def baseline_subtracted_reward(raw_residual, text_idx):
            text_idx = int(text_idx)
            raw_residual = float(raw_residual)
            if zero_residual_by_text is None or text_idx >= len(zero_residual_by_text):
                return raw_residual
            return float(max(0.0, raw_residual - float(zero_residual_by_text[text_idx])))

        if "Random" not in enabled_methods:
            print("  [Skip] Random disabled.")
            eval_Theta_arr = np.asarray([], dtype=float)
            eval_text_indices_arr = np.asarray([], dtype=int)
            residuals_all = np.asarray([], dtype=float)
            random_policy_rewards = np.asarray([], dtype=float)
        else:
            # -------------------------------------------------
            # 4.2 Evaluation phase A: random search over prompt edits with fixed w_hat
            # -------------------------------------------------
            print("  [Phase] Evaluating PIER with random prompt search (fixed budget)...")

            if len(eval_texts) == 0:
                print("  [WARN] No eval texts for this target. Skipping random-search eval.")
                continue

            eval_X = []
            eval_Theta = []
            eval_text_indices = []
            eval_seeds = []

            for step in range(search_budget):
                # Use the same deterministic text schedule as the theta-level adaptive
                # methods, so Random differs only in how it selects theta.
                text_idx = step % len(eval_texts)
                text = eval_texts[text_idx]
                theta = float(rng.choice(doses_eval))
                stable_seed = make_stable_seed(text=pair_to_text(text), theta=theta)
                eval_X.append(text)
                eval_Theta.append(theta)
                eval_text_indices.append(text_idx)
                eval_seeds.append(int(stable_seed))

            y_t_eval, Y_p_eval = eco.batched_query(
                X=eval_X,
                Thetas=eval_Theta,
                intervention=intervention,
                seeds=eval_seeds,
            )

            eval_Theta_arr = np.asarray(eval_Theta, dtype=float)
            eval_text_indices_arr = np.asarray(eval_text_indices, dtype=int)
            y_mix_eval = Y_p_eval @ w_hat
            residuals_all = np.abs(y_t_eval - y_mix_eval)
            random_policy_rewards = np.asarray(
                [
                    baseline_subtracted_reward(raw_residual, text_idx)
                    for raw_residual, text_idx in zip(residuals_all, eval_text_indices_arr)
                ],
                dtype=float,
            )

            random_best_so_far = np.maximum.accumulate(residuals_all)

            for step, (theta_step, reward_step, best_step) in enumerate(
                    zip(eval_Theta_arr, residuals_all, random_best_so_far), start=1
            ):
                all_trajectories.append({
                    "Seed": experiment_seed,
                    "Step": step,
                    "TextIndex": int(eval_text_indices_arr[step - 1]),
                    "Dose": float(theta_step),
                    "Residual": float(reward_step),
                    "PolicyReward": float(random_policy_rewards[step - 1]),
                    "BestSoFar": float(best_step),
                    "Model": t_name,
                    "Group": t_info["type"],
                    "Method": "Random",
                })

            # Optional sanity check for clone target on random-search eval
            if "clone_of_peer_idx" in t_info:
                clone_idx = t_info["clone_of_peer_idx"]
                clone_diffs = np.abs(y_t_eval - Y_p_eval[:, clone_idx])
                max_diff_eval = float(np.max(clone_diffs))
                if max_diff_eval > 1e-9:
                    print(
                        f"  [ALARM] Clone mismatch on random-search eval for {t_name}! "
                        f"Max diff: {max_diff_eval:.3e}"
                    )

            # Aggregate random-search PIER per action/dose
            for theta in doses_eval:
                mask = np.isclose(eval_Theta_arr, float(theta), atol=1e-8)
                vals = residuals_all[mask]
                count = int(np.sum(mask))
                if vals.size == 0:
                    avg_pier = float("nan")
                    best_pier = float("nan")
                else:
                    avg_pier = float(np.mean(vals))
                    best_pier = float(np.max(vals))

                all_results.append(
                    {
                        "Dose": float(theta),
                        "PIER": avg_pier,
                        "BestPIER": best_pier,
                        "Count": count,
                        "Model": t_name,
                        "Group": t_info["type"],
                        "Method": "Random",
                    }
                )




        if "JointPairUCB" not in enabled_methods:
            print("  [Skip] JointPairUCB disabled.")
            ctx_eval_Theta_arr = np.asarray([], dtype=float)
            ctx_eval_text_indices_arr = np.asarray([], dtype=int)
            ctx_rewards_arr = np.asarray([], dtype=float)
            ctx_policy_rewards_arr = np.asarray([], dtype=float)
            ctx_support_means_arr = np.asarray([], dtype=float)
        else:
            # -------------------------------------------------
            # 4.6 Diagnostic evaluation phase E: Joint UCB over full (text, prompt-edit) actions
            # -------------------------------------------------
            print("  [Phase] Evaluating PIER with JointPairUCB search over (text, prompt-edit) actions...")

            if len(eval_texts) == 0:
                print("  [WARN] No eval texts for this target. Skipping JointPairUCB eval.")
                continue

            if risk_pool_indices is not None and len(risk_pool_indices) > 0:
                candidate_text_indices = [int(i) for i in risk_pool_indices]
            else:
                candidate_text_indices = list(range(len(eval_texts)))

            # Use non-zero prompt edits for the joint prompt-regime search.
            candidate_thetas = [float(theta) for theta in doses_eval if float(theta) > 0.0]
            pair_actions = [
                (text_idx, theta)
                for text_idx in candidate_text_indices
                for theta in candidate_thetas
            ]

            prior_scores = {}
            if probe_risk_scores is not None:
                max_risk = float(np.max(probe_risk_scores)) if len(probe_risk_scores) > 0 else 1.0
                max_risk = max(max_risk, 1e-8)
                for text_idx, theta in pair_actions:
                    prior_scores[(int(text_idx), float(theta))] = float(probe_risk_scores[text_idx] / max_risk)

            joint_policy = JointPairUCBPolicy(
                pair_actions=pair_actions,
                c=joint_ucb_c,
                prior_scores=prior_scores,
            )

            ctx_eval_Theta = []
            ctx_eval_text_indices = []
            ctx_rewards = []          # raw pointwise PIER for Exp3 best-so-far reporting
            ctx_policy_rewards = []   # robust reward used to update/select prompt regimes
            ctx_support_means = []

            for step in range(search_budget):
                text_idx, theta = joint_policy.select()
                text = eval_texts[text_idx]
                stable_seed = make_stable_seed(text=pair_to_text(text), theta=theta)

                y_t_step, Y_p_step = eco.batched_query(
                    X=[text],
                    Thetas=[theta],
                    intervention=intervention,
                    seeds=[int(stable_seed)],
                )

                y_mix_step = float(Y_p_step[0] @ w_hat)
                raw_reward = float(abs(float(y_t_step[0]) - y_mix_step))
                point_policy_reward = baseline_subtracted_reward(raw_reward, text_idx)

                # Robust prompt-regime reward: after selecting (text*, theta), also
                # evaluate the same theta on a small support set of other search texts.
                # The policy is updated with a mixture of the pointwise spike and the
                # support-set mean, so it prefers prompt edits that are both high and
                # less search-text-specific.
                support_candidates = [int(i) for i in candidate_text_indices if int(i) != int(text_idx)]
                support_mean = raw_reward
                support_policy_mean = point_policy_reward
                if robust_support_size > 0 and len(support_candidates) > 0:
                    support_size = min(int(robust_support_size), len(support_candidates))
                    support_indices = rng.choice(support_candidates, size=support_size, replace=False)
                    support_X = []
                    support_Theta = []
                    support_seeds = []
                    for support_idx in support_indices:
                        support_text = eval_texts[int(support_idx)]
                        support_seed = make_stable_seed(text=pair_to_text(support_text), theta=theta)
                        support_X.append(support_text)
                        support_Theta.append(theta)
                        support_seeds.append(int(support_seed))

                    y_t_support, Y_p_support = eco.batched_query(
                        X=support_X,
                        Thetas=support_Theta,
                        intervention=intervention,
                        seeds=support_seeds,
                    )
                    y_mix_support = Y_p_support @ w_hat
                    support_residuals = np.abs(y_t_support - y_mix_support)
                    support_policy_rewards = np.asarray(
                        [
                            baseline_subtracted_reward(raw_residual, support_idx)
                            for raw_residual, support_idx in zip(support_residuals, support_indices)
                        ],
                        dtype=float,
                    )
                    support_mean = float(np.mean(support_residuals))
                    support_policy_mean = float(np.mean(support_policy_rewards))

                if robust_support_size > 0 and len(support_candidates) > 0:
                    robust_reward = (
                        float(robust_primary_weight) * point_policy_reward
                        + (1.0 - float(robust_primary_weight)) * support_policy_mean
                    )
                else:
                    robust_reward = point_policy_reward
                joint_policy.update((text_idx, theta), robust_reward)

                ctx_eval_Theta.append(theta)
                ctx_eval_text_indices.append(text_idx)
                ctx_rewards.append(raw_reward)
                ctx_policy_rewards.append(robust_reward)
                ctx_support_means.append(support_mean)

            ctx_eval_Theta_arr = np.asarray(ctx_eval_Theta, dtype=float)
            ctx_eval_text_indices_arr = np.asarray(ctx_eval_text_indices, dtype=int)
            ctx_rewards_arr = np.asarray(ctx_rewards, dtype=float)
            ctx_policy_rewards_arr = np.asarray(ctx_policy_rewards, dtype=float)
            ctx_support_means_arr = np.asarray(ctx_support_means, dtype=float)
            ctx_best_so_far = np.maximum.accumulate(ctx_rewards_arr)

            for step, (theta_step, reward_step, best_step) in enumerate(
                    zip(ctx_eval_Theta_arr, ctx_rewards_arr, ctx_best_so_far), start=1
            ):
                all_trajectories.append({
                    "Seed": experiment_seed,
                    "Step": step,
                    "TextIndex": int(ctx_eval_text_indices_arr[step - 1]),
                    "Dose": float(theta_step),
                    "Residual": float(reward_step),
                    "PolicyReward": float(ctx_policy_rewards_arr[step - 1]),
                    "SupportMeanPIER": float(ctx_support_means_arr[step - 1]),
                    "BestSoFar": float(best_step),
                    "Model": t_name,
                    "Group": t_info["type"],
                    "Method": "JointPairUCB",
                })

            # Aggregate JointPairUCB-search PIER per action/dose
            for theta in doses_eval:
                mask = np.isclose(ctx_eval_Theta_arr, float(theta), atol=1e-8)
                vals = ctx_rewards_arr[mask]
                count = int(np.sum(mask))
                if vals.size == 0:
                    avg_pier = float("nan")
                    best_pier = float("nan")
                else:
                    avg_pier = float(np.mean(vals))
                    best_pier = float(np.max(vals))

                all_results.append(
                    {
                        "Dose": float(theta),
                        "PIER": avg_pier,
                        "BestPIER": best_pier,
                        "Count": count,
                        "Model": t_name,
                        "Group": t_info["type"],
                        "Method": "JointPairUCB",
                    }
                )

        print(
            f"  [Phase] Prompt-family validation using Top-{top_k_holdout} searched "
            "(text, theta) pairs..."
        )

        method_search_records = {}
        if "Random" in enabled_methods:
            method_search_records["Random"] = (eval_Theta_arr, residuals_all, eval_text_indices_arr, random_policy_rewards)
        if "JointPairUCB" in enabled_methods:
            method_search_records["JointPairUCB"] = (ctx_eval_Theta_arr, ctx_rewards_arr, ctx_eval_text_indices_arr, ctx_policy_rewards_arr)

        for method_name, (theta_history, reward_history, text_index_history, selection_score_history) in method_search_records.items():
            # Family validation: select the best searched (text, theta) pairs, keep
            # their search text x, and validate other prompt edits from the same family.
            theta_history = np.asarray(theta_history, dtype=float)
            reward_history = np.asarray(reward_history, dtype=float)
            text_index_history = np.asarray(text_index_history, dtype=int)
            selection_score_history = np.asarray(selection_score_history, dtype=float)
            nonzero_mask = theta_history > 0.0
            if np.any(nonzero_mask):
                valid_indices = np.where(nonzero_mask)[0]
            else:
                valid_indices = np.arange(len(theta_history))

            if valid_indices.size == 0:
                continue

            # Rank candidates by the policy selection score. Family mode uses this
            # to choose the strongest searched (text, theta) pairs; theta mode uses
            # it to choose the strongest theta candidates for transfer.
            ranked_valid_indices = valid_indices[np.argsort(-selection_score_history[valid_indices])]

            # Exp3 search remains max-based: keep the single best pair for reporting.
            best_global_idx = int(valid_indices[np.nanargmax(reward_history[valid_indices])])
            theta_star = float(theta_history[best_global_idx])
            search_best_pier = float(reward_history[best_global_idx])
            search_pair_index = int(best_global_idx)
            search_text_index = int(text_index_history[best_global_idx])

            # Prompt-family validation:
            # 1) take Top-K searched (text, theta) pairs;
            # 2) keep the same searched text x;
            # 3) replace theta by other prompt edits from the same family;
            # 4) measure whether the discovered prompt mechanism is stable.
            top_pair_indices = []
            seen_pair_keys = set()
            for idx in ranked_valid_indices:
                theta_candidate = float(theta_history[idx])
                if theta_candidate <= 0.0:
                    continue
                text_idx_candidate = int(text_index_history[idx])
                family_candidate = prompt_family(theta_candidate)
                pair_key = (text_idx_candidate, family_candidate)
                if pair_key in seen_pair_keys:
                    continue
                top_pair_indices.append(int(idx))
                seen_pair_keys.add(pair_key)
                if len(top_pair_indices) >= top_k_holdout:
                    break

            if not top_pair_indices:
                continue

            family_X = []
            family_Theta = []
            family_seeds = []
            family_meta = []
            pair_slices = []
            cursor = 0

            for pair_rank, idx in enumerate(top_pair_indices, start=1):
                source_text_idx = int(text_index_history[idx])
                source_text = eval_texts[source_text_idx]
                source_theta = float(theta_history[idx])
                source_family = prompt_family(source_theta)

                same_family_thetas = [
                    float(theta)
                    for theta in doses_eval
                    if float(theta) > 0.0
                    and prompt_family(theta) == source_family
                    and not np.isclose(float(theta), source_theta, atol=1e-8)
                ]
                if not same_family_thetas:
                    same_family_thetas = [source_theta]

                start_cursor = cursor
                for theta_candidate in same_family_thetas:
                    stable_seed = make_stable_seed(text=pair_to_text(source_text), theta=theta_candidate)
                    family_X.append(source_text)
                    family_Theta.append(theta_candidate)
                    family_seeds.append(int(stable_seed))
                    family_meta.append(
                        {
                            "pair_rank": int(pair_rank),
                            "source_text_idx": int(source_text_idx),
                            "source_theta": float(source_theta),
                            "source_family": str(source_family),
                            "theta_candidate": float(theta_candidate),
                        }
                    )
                    cursor += 1
                pair_slices.append((start_cursor, cursor))

            y_t_family, Y_p_family = eco.batched_query(
                X=family_X,
                Thetas=family_Theta,
                intervention=intervention,
                seeds=family_seeds,
            )

            y_mix_family = Y_p_family @ w_hat
            family_residuals_all = np.abs(y_t_family - y_mix_family)

            for flat_idx, meta in enumerate(family_meta):
                all_holdout_pointwise_results.append({
                    "Seed": experiment_seed,
                    "HoldoutTextIndex": int(meta["source_text_idx"]),
                    "Dose": float(meta["theta_candidate"]),
                    "ThetaRank": int(meta["pair_rank"]),
                    "Residual": float(family_residuals_all[flat_idx]),
                    "TargetOutput": float(y_t_family[flat_idx]),
                    "PeerMixOutput": float(y_mix_family[flat_idx]),
                    "Model": t_name,
                    "Group": t_info["type"],
                    "Method": method_name,
                    "SelectedTopKDoses": ";".join(str(float(theta_history[idx])) for idx in top_pair_indices),
                    "PointwiseType": "family_theta_specific",
                    "ValidationMode": "family",
                    "SourceTextIndex": int(meta["source_text_idx"]),
                    "SourceDose": float(meta["source_theta"]),
                    "PromptFamily": str(meta["source_family"]),
                })

            family_pair_max_residuals = []
            family_pair_argmax_thetas = []
            for pair_rank, (start_cursor, end_cursor) in enumerate(pair_slices, start=1):
                vals = family_residuals_all[start_cursor:end_cursor]
                local_argmax = int(np.argmax(vals))
                best_flat_idx = start_cursor + local_argmax
                best_meta = family_meta[best_flat_idx]
                best_residual = float(vals[local_argmax])
                family_pair_max_residuals.append(best_residual)
                family_pair_argmax_thetas.append(float(best_meta["theta_candidate"]))
                all_holdout_pointwise_results.append({
                    "Seed": experiment_seed,
                    "HoldoutTextIndex": int(best_meta["source_text_idx"]),
                    "Dose": float(best_meta["theta_candidate"]),
                    "ThetaRank": int(pair_rank),
                    "Residual": best_residual,
                    "TargetOutput": float("nan"),
                    "PeerMixOutput": float("nan"),
                    "Model": t_name,
                    "Group": t_info["type"],
                    "Method": method_name,
                    "SelectedTopKDoses": ";".join(str(float(theta_history[idx])) for idx in top_pair_indices),
                    "PointwiseType": "family_max_per_pair",
                    "ValidationMode": "family",
                    "SourceTextIndex": int(best_meta["source_text_idx"]),
                    "SourceDose": float(best_meta["source_theta"]),
                    "PromptFamily": str(best_meta["source_family"]),
                })

            family_pair_max_residuals = np.asarray(family_pair_max_residuals, dtype=float)
            sorted_family = np.sort(family_pair_max_residuals)[::-1]
            top_tail_k = max(1, int(np.ceil(0.10 * len(sorted_family))))
            search_best_pair_idx = int(top_pair_indices[0])

            all_holdout_results.append({
                "Seed": experiment_seed,
                "SelectedDose": float(theta_history[search_best_pair_idx]),
                "SelectedTopKDoses": ";".join(str(float(theta_history[idx])) for idx in top_pair_indices),
                "TopK": int(len(top_pair_indices)),
                "SearchBestPIER": float(reward_history[search_best_pair_idx]),
                "SearchBestSelectionScore": float(selection_score_history[search_best_pair_idx]),
                "SearchPairIndex": int(search_best_pair_idx),
                "SearchTextIndex": int(text_index_history[search_best_pair_idx]),
                "HoldoutPIER": float(np.mean(family_pair_max_residuals)),
                "HoldoutMeanOverTopKPIER": float(np.mean(family_residuals_all)),
                "HoldoutBestPIER": float(np.max(family_residuals_all)),
                "HoldoutQ90PIER": float(np.quantile(family_pair_max_residuals, 0.90)),
                "HoldoutQ95PIER": float(np.quantile(family_pair_max_residuals, 0.95)),
                "HoldoutTop10MeanPIER": float(np.mean(sorted_family[:top_tail_k])),
                "HoldoutStdPIER": float(np.std(family_pair_max_residuals)),
                "HoldoutCount": int(len(family_pair_max_residuals)),
                "Model": t_name,
                "Group": t_info["type"],
                "Method": method_name,
                "ActionDefinitionNote": "prompt-family validation; fixed search text x, swap theta within same family",
                "ValidationMode": "family",
                "SelectedPromptFamilies": ";".join(prompt_family(theta_history[idx]) for idx in top_pair_indices),
                "FamilyBestDoses": ";".join(str(theta) for theta in family_pair_argmax_thetas),
            })
            # continue  # Remove the continue


    # =====================================================================
    # 5. Save results
    # =====================================================================
    output_dir = "results/thesis_exp4/mnli"
    os.makedirs(output_dir, exist_ok=True)

    suffix = "" if top_k_holdout == TOP_K_HOLDOUT else f"_top{top_k_holdout}"
    suffix = f"{suffix}_family"

    df = pd.DataFrame(all_results)
    out_path = os.path.join(output_dir, f"exp23_mnli_prompt_bandit_dosesplit{suffix}_seed{experiment_seed}{run_tag_suffix}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved aggregate results to: {out_path}")

    traj_df = pd.DataFrame(all_trajectories)
    traj_path = os.path.join(output_dir, f"exp23_mnli_prompt_bandit_trajectory{suffix}_seed{experiment_seed}{run_tag_suffix}.csv")
    traj_df.to_csv(traj_path, index=False)
    print(f"Saved search trajectory results to: {traj_path}")

    holdout_df = pd.DataFrame(all_holdout_results)
    holdout_path = os.path.join(output_dir, f"exp23_mnli_prompt_bandit_holdout{suffix}_seed{experiment_seed}{run_tag_suffix}.csv")
    holdout_df.to_csv(holdout_path, index=False)
    print(f"Saved holdout results to: {holdout_path}")

    holdout_pointwise_df = pd.DataFrame(all_holdout_pointwise_results)
    holdout_pointwise_path = os.path.join(
        output_dir,
        f"exp23_mnli_prompt_bandit_holdout_pointwise{suffix}_seed{experiment_seed}{run_tag_suffix}.csv",
    )
    holdout_pointwise_df.to_csv(holdout_pointwise_path, index=False)
    print(f"Saved pointwise holdout residuals to: {holdout_pointwise_path}")

    print("Done.")


def parse_seed_list(seed_arg):
    """Parse --seed as either a single integer or a comma-separated list, e.g. 0,1,2,3,4."""
    if isinstance(seed_arg, int):
        return [seed_arg]
    seeds = []
    for part in str(seed_arg).split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("--seed must contain at least one integer seed.")
    return seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=str,
        default="0",
        help="Single seed or comma-separated seeds, e.g. 0 or 0,1,2,3,4.",
    )
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--search_budget", type=int, default=100)
    parser.add_argument(
        "--top_k_holdout",
        type=int,
        default=TOP_K_HOLDOUT,
        help="Number of searched pairs/prompts transferred to validation. Use 1 for strict Top-1 family validation.",
    )
    parser.add_argument(
        "--holdout_mode",
        type=str,
        default="family",
        choices=["family"],
        help="Family validation only: fixed searched text, swap theta within the same prompt family.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated methods to run, e.g. Random or Random,JointPairUCB. Use all for every method.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix for output filenames.",
    )
    args = parser.parse_args()

    seed_list = parse_seed_list(args.seed)
    for seed in seed_list:
        run_bert_experiment(
            seed=seed,
            max_samples=args.max_samples,
            search_budget=args.search_budget,
            top_k_holdout=args.top_k_holdout,
            holdout_mode=args.holdout_mode,
            methods=args.methods,
            run_tag=args.run_tag,
        )
