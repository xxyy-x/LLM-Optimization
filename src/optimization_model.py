"""
Two-Stage Stochastic Budget-Constrained LLM Routing Model
==========================================================

Stage 1  (strategic, before observing prompts):
    y_m ∈ {0,1}  — select a pool of at most K models

Stage 2  (operational, after observing each prompt p):
    x_{pm} ∈ {0,1} — route prompt p to one model in the pool

Objective  (maximise):
    Σ_p w_p Σ_m s_{pm} x_{pm}  -  λ Σ_d slack_d

Constraints:
    (1) Pool size       : Σ_m y_m ≤ K
    (2) Assignment      : Σ_m x_{pm} = 1            ∀ p
    (3) Routing linkage : x_{pm} ≤ y_m               ∀ p, m
    (4) Budget          : Σ_p w_p Σ_m c_{pm} x_{pm} ≤ B
    (5) Quality (soft)  : avg_score(d) ≥ Q_d − slack_d   ∀ d ∈ D

The SAA interpretation: w_p approximates the probability that a random
incoming prompt is of type p.  Adjusting dataset multipliers lets you
model different company focus areas (coding, math, knowledge, balanced).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pyomo.environ as pyo

from .data_loading import make_parameter_dicts
from .util import choose_solver


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------

def build_weighted_prompt_weights(
    df: pd.DataFrame,
    dataset_multipliers: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Build SAA prompt weights w_p with optional dataset emphasis.

    Each prompt in a dataset receives a raw weight equal to the dataset's
    multiplier.  Weights are then normalised so that Σ_p w_p = 1.

    Parameters
    ----------
    df : DataFrame
        RouterBench data (must have columns ``prompt_id``, ``dataset``).
    dataset_multipliers : dict or None
        Relative multiplier per dataset, e.g. ``{'LCB': 4.0}``.
        Datasets not listed default to 1.0.

    Returns
    -------
    dict  {prompt_id → weight}

    Examples
    --------
    Uniform (default):
        build_weighted_prompt_weights(df)

    Coding-focused company (Cursor-like):
        build_weighted_prompt_weights(df, {'LCB': 4.0})

    Math-focused tutoring platform:
        build_weighted_prompt_weights(df, {'AIME': 4.0})

    Knowledge-focused (GPQA + MMLU-Pro):
        build_weighted_prompt_weights(df, {'GPQA': 3.0, 'MMLU-Pro': 3.0})
    """
    if dataset_multipliers is None:
        dataset_multipliers = {}

    prompt_dataset: dict[str, str] = (
        df[["prompt_id", "dataset"]]
        .drop_duplicates()
        .set_index("prompt_id")["dataset"]
        .to_dict()
    )

    raw: dict[str, float] = {
        p: dataset_multipliers.get(d, 1.0)
        for p, d in prompt_dataset.items()
    }

    total = sum(raw.values())
    return {p: w / total for p, w in raw.items()}


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_two_stage_router(
    df: pd.DataFrame,
    prompt_weights: dict[str, float],
    *,
    K: int = 5,
    B: float = 0.01,
    Q_min_by_dataset: Optional[dict[str, float]] = None,
    slack_penalty: float = 10.0,
    verbose: bool = False,
) -> dict:
    """
    Solve the two-stage stochastic LLM routing optimisation model.

    Hyperparameters
    ---------------
    K : int
        Maximum number of models in the selected pool  (Stage-1 cardinality).
    B : float
        Budget — maximum weighted-average cost per prompt (dollars).
        Set B = float('inf') to disable the budget constraint.
    Q_min_by_dataset : dict or None
        Minimum average score required per benchmark dataset.
        e.g. ``{'LCB': 0.70, 'AIME': 0.50}``.
        Datasets not listed default to 0 (inactive constraint).
        Violations are allowed but penalised via ``slack_penalty``.
    slack_penalty : float
        Penalty coefficient λ per unit of quality-constraint violation.
        Higher values enforce quality constraints more strictly.
    verbose : bool
        Whether to print solver log.

    Returns
    -------
    dict with keys:
        selected_models      : list[str]   — names of chosen models
        assignments          : DataFrame   — per-prompt routing decisions
        avg_cost             : float       — weighted average cost
        avg_score            : float       — weighted average score
        slack_by_dataset     : dict        — quality violation per dataset
        solver_status        : str
        termination_condition: str
        params               : dict        — hyperparameters used
    """
    if Q_min_by_dataset is None:
        Q_min_by_dataset = {}

    prompts, models, datasets, score, cost, dataset_of_prompt = make_parameter_dicts(df)
    available_pairs = sorted(score.keys())

    prompts_by_dataset: dict[str, list[str]] = {
        d: [p for p in prompts if dataset_of_prompt[p] == d]
        for d in datasets
    }

    # ------------------------------------------------------------------
    # Build Pyomo model
    # ------------------------------------------------------------------
    mdl = pyo.ConcreteModel()

    # Sets
    mdl.P  = pyo.Set(initialize=prompts)
    mdl.M  = pyo.Set(initialize=models)
    mdl.PM = pyo.Set(initialize=available_pairs, dimen=2)
    mdl.D  = pyo.Set(initialize=datasets)

    # Parameters
    mdl.w     = pyo.Param(mdl.P,  initialize=prompt_weights,                                   within=pyo.NonNegativeReals)
    mdl.s     = pyo.Param(mdl.PM, initialize=score,                                             within=pyo.Reals)
    mdl.c     = pyo.Param(mdl.PM, initialize=cost,                                              within=pyo.NonNegativeReals)
    mdl.Q_min = pyo.Param(mdl.D,  initialize={d: Q_min_by_dataset.get(d, 0.0) for d in datasets},
                           within=pyo.NonNegativeReals)
    mdl.lam   = pyo.Param(initialize=slack_penalty, within=pyo.NonNegativeReals)

    # --- Stage-1: pool selection ---
    mdl.y = pyo.Var(mdl.M, within=pyo.Binary)

    # --- Stage-2: routing ---
    mdl.x = pyo.Var(mdl.PM, within=pyo.Binary)

    # --- Slack for soft quality guarantees ---
    # Named qual_slack to avoid collision with Pyomo's internal 'slack' attribute
    mdl.qual_slack = pyo.Var(mdl.D, within=pyo.NonNegativeReals)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # (1) Pool size
    mdl.pool_size = pyo.Constraint(
        expr=sum(mdl.y[m] for m in mdl.M) <= K
    )

    # (2) Assignment — each prompt routed to exactly one model
    def _assignment(m, p):
        feasible = [mm for (pp, mm) in m.PM if pp == p]
        return sum(m.x[p, mm] for mm in feasible) == 1

    mdl.assignment = pyo.Constraint(mdl.P, rule=_assignment)

    # (3) Routing linkage — only route to selected models
    def _linkage(m, p, mm):
        return m.x[p, mm] <= m.y[mm]

    mdl.linkage = pyo.Constraint(mdl.PM, rule=_linkage)

    # (4) Budget
    if B < float("inf"):
        mdl.budget = pyo.Constraint(
            expr=sum(mdl.w[p] * mdl.c[p, mm] * mdl.x[p, mm]
                     for (p, mm) in mdl.PM) <= B
        )

    # (5) Per-dataset soft quality guarantee
    def _quality(m, d):
        pd_set = set(prompts_by_dataset[d])
        n_d    = len(prompts_by_dataset[d])
        avg_d  = sum(
            m.s[p, mm] * m.x[p, mm]
            for (p, mm) in m.PM if p in pd_set
        ) / n_d
        return avg_d >= m.Q_min[d] - m.qual_slack[d]

    mdl.quality = pyo.Constraint(mdl.D, rule=_quality)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _objective(m):
        avg_score = sum(m.w[p] * m.s[p, mm] * m.x[p, mm] for (p, mm) in m.PM)
        penalty   = sum(m.lam * m.qual_slack[d] for d in m.D)
        return avg_score - penalty

    mdl.obj = pyo.Objective(rule=_objective, sense=pyo.maximize)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver  = choose_solver()
    results = solver.solve(mdl, tee=verbose)

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    selected_models = sorted(
        m for m in models if pyo.value(mdl.y[m]) > 0.5
    )

    chosen_rows = [
        {
            "prompt_id": p,
            "model":     mm,
            "dataset":   dataset_of_prompt[p],
            "weight":    prompt_weights[p],
            "score":     score[(p, mm)],
            "cost":      cost[(p, mm)],
        }
        for (p, mm) in mdl.PM
        if pyo.value(mdl.x[p, mm]) > 0.5
    ]

    assignments = (
        pd.DataFrame(chosen_rows)
        .sort_values(["dataset", "prompt_id"])
        .reset_index(drop=True)
    )

    avg_cost  = float((assignments["weight"] * assignments["cost"]).sum())
    avg_score = float((assignments["weight"] * assignments["score"]).sum())

    return {
        "selected_models":       selected_models,
        "assignments":           assignments,
        "avg_cost":              avg_cost,
        "avg_score":             avg_score,
        "slack_by_dataset":      {d: float(pyo.value(mdl.qual_slack[d])) for d in datasets},
        "solver_status":         str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "params": {
            "K":                 K,
            "B":                 B,
            "Q_min_by_dataset":  Q_min_by_dataset,
            "slack_penalty":     slack_penalty,
        },
    }


# ---------------------------------------------------------------------------
# Q1 — Pool size sweep
# ---------------------------------------------------------------------------

def sweep_pool_size(
    df: pd.DataFrame,
    prompt_weights: dict[str, float],
    K_values: Optional[list[int]] = None,
    *,
    B: float = 0.01,
    Q_min_by_dataset: Optional[dict[str, float]] = None,
    slack_penalty: float = 10.0,
) -> pd.DataFrame:
    """
    Q1 Analysis: how large must the model pool be?

    Sweeps over values of K (pool size) and records the optimal average
    score and cost for each.  Useful for plotting diminishing-returns curves.

    Parameters
    ----------
    K_values : list[int] or None
        Pool sizes to evaluate.  Defaults to 1 … 10.

    Returns
    -------
    DataFrame with columns:
        K, avg_score, avg_cost, selected_models, n_selected, slack_by_dataset
    """
    if K_values is None:
        K_values = list(range(1, 11))

    records = []
    for K in K_values:
        print(f"  Solving K = {K} ...", end="  ")
        res = solve_two_stage_router(
            df, prompt_weights,
            K=K, B=B,
            Q_min_by_dataset=Q_min_by_dataset,
            slack_penalty=slack_penalty,
        )
        print(
            f"avg_score = {res['avg_score']:.4f},  "
            f"avg_cost = {res['avg_cost']:.6f},  "
            f"pool = {res['selected_models']}"
        )
        records.append({
            "K":                K,
            "avg_score":        res["avg_score"],
            "avg_cost":         res["avg_cost"],
            "selected_models":  res["selected_models"],
            "n_selected":       len(res["selected_models"]),
            "slack_by_dataset": res["slack_by_dataset"],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Q2 — Prompt-weight scenario sweep
# ---------------------------------------------------------------------------

def sweep_prompt_weights(
    df: pd.DataFrame,
    weight_scenarios: dict[str, dict[str, float]],
    *,
    K: int = 5,
    B: float = 0.01,
    Q_min_by_dataset: Optional[dict[str, float]] = None,
    slack_penalty: float = 10.0,
) -> pd.DataFrame:
    """
    Q2 Analysis: how does the optimal pool change under different prompt
    weight distributions? (SAA robustness / distribution-shift study)

    Each scenario re-weights prompts by dataset to simulate a different
    company focus (coding, math, knowledge, balanced).

    Parameters
    ----------
    weight_scenarios : dict
        Mapping  scenario_name → dataset_multipliers.
        Example::

            {
                'Balanced':             {},
                'Coding (LCB ×4)':      {'LCB':    4.0},
                'Math (AIME ×4)':       {'AIME':   4.0},
                'Knowledge (×4)':       {'GPQA': 3.0, 'MMLU-Pro': 3.0},
            }

    Returns
    -------
    DataFrame with columns:
        scenario, avg_score, avg_cost, selected_models, n_selected,
        slack_by_dataset, multipliers
    """
    records = []
    for name, multipliers in weight_scenarios.items():
        print(f"  Scenario: {name!r} ...", end="  ")
        w   = build_weighted_prompt_weights(df, multipliers)
        res = solve_two_stage_router(
            df, w,
            K=K, B=B,
            Q_min_by_dataset=Q_min_by_dataset,
            slack_penalty=slack_penalty,
        )
        print(
            f"avg_score = {res['avg_score']:.4f},  "
            f"avg_cost = {res['avg_cost']:.6f}"
        )
        records.append({
            "scenario":          name,
            "multipliers":       multipliers,
            "avg_score":         res["avg_score"],
            "avg_cost":          res["avg_cost"],
            "selected_models":   res["selected_models"],
            "n_selected":        len(res["selected_models"]),
            "slack_by_dataset":  res["slack_by_dataset"],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cross-domain evaluation
# ---------------------------------------------------------------------------

def evaluate_pool_greedy(
    df: pd.DataFrame,
    selected_models: list,
    target_dataset: Optional[str] = None,
) -> dict:
    """
    Evaluate a fixed model pool using greedy Stage-2 routing (no solver needed).

    For each prompt, assigns the model in the pool with the highest score;
    ties are broken by lowest cost.  No budget constraint is applied — this
    is a pure performance upper-bound given the pool.

    Parameters
    ----------
    selected_models : list[str]
        Model names in the fixed pool.
    target_dataset : str or None
        If given, only evaluate prompts from this dataset.

    Returns
    -------
    dict with keys:
        avg_score            : float  — overall weighted average score
        avg_score_by_dataset : dict   — per-dataset average score
        assignments          : DataFrame
    """
    df_eval = df[df["model"].isin(selected_models)].copy()
    if target_dataset:
        df_eval = df_eval[df_eval["dataset"] == target_dataset]

    best = (
        df_eval
        .sort_values(["prompt_id", "score", "cost"], ascending=[True, False, True])
        .groupby("prompt_id", as_index=False)
        .first()
    )

    per_dataset = best.groupby("dataset")["score"].mean().to_dict()
    overall     = float(best["score"].mean())

    return {
        "avg_score":            overall,
        "avg_score_by_dataset": per_dataset,
        "assignments":          best,
    }


def cross_domain_matrix(
    df: pd.DataFrame,
    pure_pools: dict,
) -> pd.DataFrame:
    """
    Build the cross-domain performance matrix.

    Parameters
    ----------
    pure_pools : dict
        {train_dataset -> list of selected_models}
        Typically the result of solving with pure-specialization weights.

    Returns
    -------
    DataFrame  — rows = optimised-for dataset, columns = evaluated-on dataset.
    Cell [i, j] = avg score of pool i when applied to dataset j.
    The diagonal = in-distribution performance; off-diagonal = OOD performance.
    """
    datasets = sorted(df["dataset"].unique().tolist())

    rows = []
    for train_d, pool in pure_pools.items():
        scores = {}
        for test_d in datasets:
            res = evaluate_pool_greedy(df, pool, target_dataset=test_d)
            scores[test_d] = res["avg_score_by_dataset"].get(test_d, float("nan"))
        rows.append({"optimised_for": train_d, **scores})

    # Reindex rows to match column (alphabetical) order so the diagonal is aligned
    return pd.DataFrame(rows).set_index("optimised_for")[datasets].reindex(datasets)

    return pd.DataFrame(records)
