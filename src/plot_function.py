import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .data_loading import *
from .util import *
import seaborn as sns

def plot_frontier(results_df: pd.DataFrame, output_path: Path):
    """
    Scatter/line plot on the cost-performance plane.
    x-axis: average cost (weighted by prompt-weights)
    y-axis: average score (weighted by prompt-weights)
    """
    plt.figure(figsize=(10, 7))

    # Oracle weighted-sum routing frontier
    opt = results_df[results_df["policy_name"] == "Oracle Routing"].sort_values("alpha")
    plt.plot(opt["avg_cost"], opt["avg_score"], marker="o", linewidth=2, label="Oracle Routing")

    # for _, row in opt.iterrows():
    #     plt.annotate(
    #         f"a={row['alpha']}",
    #         (row["avg_cost"], row["avg_score"]),
    #         textcoords="offset points",
    #         xytext=(5, 5),
    #         fontsize=8
    #     )

    # Single Best baseline
    sb = results_df[results_df["policy_name"] == "Single Best"].sort_values("alpha")
    plt.plot(sb["avg_cost"], sb["avg_score"], marker="s", linestyle="--", linewidth=1.5, label="Single Best")

    # Single Best per Benchmark baseline
    sbb = results_df[results_df["policy_name"] == "Single Best per Benchmark"].sort_values("alpha")
    plt.plot(
        sbb["avg_cost"], sbb["avg_score"],
        marker="^", linestyle="--", linewidth=1.5,
        label="Single Best per Benchmark"
    )

    '''
    ADD YOUR SOLUTION HERE TO BE PLOTTED ON THE SAME GRAPH
    '''

    plt.xlabel("Weighted Average Cost")
    plt.ylabel("Weighted Average Performance (Score)")
    plt.title("Routing Policies on the Cost-Performance Plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()