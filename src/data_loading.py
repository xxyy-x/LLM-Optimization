from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from .util import *

def load_routerbench_data(csv_path: str) -> pd.DataFrame:
    """
    Load the consolidated prompt-model dataset.

    Each row corresponds to a single prompt-model pair.

    Explanation of column fields:
      dataset: name of the benchmark dataset the prompt is sampled from.

      prompt_id: unique identifier of the prompt in the original benchmark.

      model: name of the model used to evaluate the prompt.

      score: binary performance score of the model on the prompt (0 or 1).

      cost: cost of evaluating the prompt (in dollar); if cost is zero, it means the model is open-source and can be evaluated locally without a cost-generating API call.
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        "dataset", "prompt_id", "model", "score", "cost"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Normalize types
    df["dataset"] = df["dataset"].astype(str)
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["model"] = df["model"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)

    # Keep one row per (prompt_id, model)
    dupes = df.duplicated(subset=["prompt_id", "model"])
    if dupes.any():
        print(
            f"Warning: found {dupes.sum()} duplicated (prompt_id, model) rows. "
            "Keeping the first occurrence."
        )
        df = df.drop_duplicates(subset=["prompt_id", "model"], keep="first").copy()

    return df


def build_prompt_weights(df: pd.DataFrame, equal_benchmark_weights: bool = True):
    """
    Construct prompt weights w_p for sample average approximation (SAA).

    By default, this function gets each prompt equal weight globally.

    You may modify this function to reflect a desired prompt distribution.
    """
    prompt_dataset = (
        df[["prompt_id", "dataset"]]
        .drop_duplicates()
        .set_index("prompt_id")["dataset"]
        .to_dict()
    )

    prompts = sorted(prompt_dataset.keys())
    datasets = sorted(set(prompt_dataset.values()))

    w = {p: 1.0 / len(prompts) for p in prompts}
    return w


def make_parameter_dicts(df: pd.DataFrame):
    """
    Create dictionaries score[(p,m)] and cost[(p,m)] for Pyomo.
    Assumes one row per (prompt_id, model).
    """
    score = {}
    cost = {}
    dataset_of_prompt = {}

    for _, row in df.iterrows():
        p = row["prompt_id"]
        m = row["model"]
        d = row["dataset"]
        score[(p, m)] = safe_float(row["score"], 0.0)
        cost[(p, m)] = safe_float(row["cost"], 0.0)
        dataset_of_prompt[p] = d

    prompts = sorted(df["prompt_id"].unique().tolist())
    models = sorted(df["model"].unique().tolist())
    datasets = sorted(df["dataset"].unique().tolist())

    return prompts, models, datasets, score, cost, dataset_of_prompt