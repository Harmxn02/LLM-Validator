"""Plots for the batch experiment results: convergence trajectories, per-model
spread, and cost/quality/characteristics views.

Usage:
    uv run python -m analysis.plots --results-csv results/experiments.csv --output-dir plots
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import _final_iteration_per_run, build_trajectory
from util.results_store import load_results

# One shared theme for every figure in this module, so a reader flipping
# between plots sees one consistent visual system rather than a new style
# per chart.
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "0.3",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    }
)

# Static metadata about each local model, since it isn't derivable from the
# results CSV. Feed this into plot_cost_quality() / the model-size plots.
# billion_params is total (on-disk) parameter count; "e4b"-style Gemma names
# denote an effective active-parameter subset of a larger MatFormer network,
# but the on-disk count is used here for an apples-to-apples x-axis. Models
# missing from this dict are silently dropped from the plots that need it.
MODEL_INFO = {
    "smollm2:135m": {"reasoning": False, "billion_params": 0.135},
    "gemma3:1b": {"reasoning": False, "billion_params": 1.0},
    "gemma4:e4b": {"reasoning": False, "billion_params": 8.0},
    "qwen3:8b": {"reasoning": True, "billion_params": 8.19},
    "qwen2.5-coder:14b": {"reasoning": False, "billion_params": 14.8},
    "qwen3.5:9b": {"reasoning": True, "billion_params": 9.65},
}

REASONING_PALETTE = {"Reasoning": "#d95f02", "No reasoning": "#1b9e77"}


def _model_palette(models: list[str]) -> dict:
    """One stable colour per model (tab10), shared across every plot that
    breaks results down by model so the same model always reads as the same
    colour."""
    models = sorted(models)
    return dict(zip(models, sns.color_palette("tab10", n_colors=len(models))))


def _with_model_info(per_model: pd.DataFrame) -> pd.DataFrame:
    info = pd.DataFrame.from_dict(MODEL_INFO, orient="index").reset_index(names="model")
    merged = per_model.merge(info, on="model", how="inner")
    merged["reasoning_label"] = merged["reasoning"].map(
        {True: "Reasoning", False: "No reasoning"}
    )
    return merged


def plot_error_trajectory(df: pd.DataFrame, output_path: str, metric: str) -> None:
    """Mean metric vs reprompt iteration, one coloured line per model, with the
    feedback and blind conditions shown as side-by-side panels sharing a
    y-axis so the two correction strategies are directly comparable. Shaded
    SEM bands replace errorbars for a calmer read when lines overlap."""
    conditions = ["feedback", "blind"]
    models = sorted(df["model"].unique())
    colours = _model_palette(models)

    fig, axes = plt.subplots(1, len(conditions), figsize=(11, 5), sharey=True)

    for ax, condition in zip(axes, conditions):
        pivot = build_trajectory(df, condition, metric)
        for model in models:
            sub = pivot.xs(model, level="model")
            means = sub.mean(axis=0)
            sems = sub.sem(axis=0)
            ax.plot(
                means.index,
                means.values,
                marker="o",
                markersize=5.5,
                linewidth=2,
                label=model,
                color=colours[model],
            )
            ax.fill_between(
                means.index,
                means.values - sems.values,
                means.values + sems.values,
                color=colours[model],
                alpha=0.18,
                linewidth=0,
            )
        ax.set_xlabel("Reprompt iteration (0 = initial generation)")
        ax.set_title(f"{condition.capitalize()} condition")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(metric.capitalize())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.suptitle(f"{metric.capitalize()} vs reprompt iteration", y=1.14, fontsize=15)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_distributions(df: pd.DataFrame, output_path: str, metric: str) -> None:
    """Final metric distribution per model, one panel per condition (mirroring
    the trajectory plot's layout) rather than 3 hues dodged onto a single
    shared x-axis — cramming (models x conditions) boxes into one axis is
    what was squishing every box down to a sliver. Each panel now only holds
    one box per model, coloured with the same per-model palette used
    everywhere else, with a jittered strip overlay showing every trial."""
    conditions = ["initial", "feedback", "blind"]
    models = sorted(df["model"].unique())
    palette = _model_palette(models)

    rows = []
    for condition in conditions:
        sub = (
            df[df["condition"] == "initial"]
            if condition == "initial"
            else _final_iteration_per_run(df, condition)
        )
        sub = sub[["model", metric]].copy()
        sub["condition"] = condition
        rows.append(sub)
    long_df = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, len(conditions), figsize=(13, 5.5), sharey=True)
    for ax, condition in zip(axes, conditions):
        sub = long_df[long_df["condition"] == condition]
        sns.boxplot(
            data=sub,
            x="model",
            y=metric,
            order=models,
            hue="model",
            palette=palette,
            legend=False,
            width=0.55,
            fliersize=0,
            linewidth=1.3,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="model",
            y=metric,
            order=models,
            color="black",
            alpha=0.45,
            size=4,
            jitter=0.15,
            ax=ax,
        )
        ax.set_title(f"{condition.capitalize()} condition")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(alpha=0.3, axis="y")

    # Error/warning counts are heavily right-skewed (mostly 0, with rare large
    # outliers), which flattens every box into an invisible line on a linear
    # axis. symlog keeps 0 representable (unlike a pure log axis) while still
    # giving the near-zero bulk of the data room to spread out.
    if long_df[metric].max() > 10:
        axes[0].set_yscale("symlog", linthresh=1)
        axes[0].set_ylim(bottom=0)

    axes[0].set_ylabel(f"Final {metric}")
    fig.suptitle(f"Final {metric} distribution by model and condition", y=1.05, fontsize=15)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _per_model_cost_quality(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean total generation time (initial + feedback reprompts) per run and
    mean final metric, aggregated to one row per model. Shared by the
    cost/quality plot and the model-size plots so both read from a single
    source of truth."""
    finals = _final_iteration_per_run(df, "feedback")[["run_id", "model", metric]]
    initial_time = df[df["condition"] == "initial"][["run_id", "gen_time_s"]].rename(
        columns={"gen_time_s": "initial_time"}
    )
    feedback_time = (
        df[df["condition"] == "feedback"]
        .groupby("run_id")["gen_time_s"]
        .sum()
        .rename("feedback_time")
    )

    merged = finals.merge(initial_time, on="run_id").merge(feedback_time, on="run_id")
    merged["total_time_s"] = merged["initial_time"] + merged["feedback_time"]

    return (
        merged.groupby("model")
        .agg(mean_time=("total_time_s", "mean"), mean_metric=(metric, "mean"))
        .reset_index()
    )


def plot_cost_quality(df: pd.DataFrame, output_path: str, metric: str) -> None:
    """Mean total generation time (initial + feedback reprompts) per run vs
    mean final error count, one point per model — a faster model that
    self-corrects worse is a real tradeoff, not a strictly worse choice.
    Colour encodes reasoning capacity and point size encodes parameter count,
    so all three axes of "which model should I use" (speed, quality, cost of
    running it) show up in one figure. A grey dashed trend line (manual
    least-squares fit) replaces the old regplot, since regplot can't drive
    hue/size mappings itself."""
    per_model = _with_model_info(_per_model_cost_quality(df, metric))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if len(per_model) >= 2:
        coeffs = np.polyfit(per_model["mean_time"], per_model["mean_metric"], 1)
        xs = np.linspace(per_model["mean_time"].min(), per_model["mean_time"].max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="0.6", linestyle="--", linewidth=1.5, zorder=1)

    sns.scatterplot(
        data=per_model,
        x="mean_time",
        y="mean_metric",
        hue="reasoning_label",
        palette=REASONING_PALETTE,
        size="billion_params",
        sizes=(90, 550),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
        zorder=2,
    )
    for _, row in per_model.iterrows():
        ax.annotate(
            row["model"],
            (row["mean_time"], row["mean_metric"]),
            fontsize=9,
            xytext=(8, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean total generation time per run (s)")
    ax.set_ylabel(f"Mean final {metric} (feedback condition)")
    ax.set_title("Cost vs quality tradeoff")
    ax.grid(alpha=0.3)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_model_size_scatter(
    df: pd.DataFrame, output_path: str, metric: str, y_col: str, y_label: str, title: str
) -> None:
    per_model = _with_model_info(_per_model_cost_quality(df, metric))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=per_model,
        x="billion_params",
        y=y_col,
        hue="reasoning_label",
        style="reasoning_label",
        palette=REASONING_PALETTE,
        s=180,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    for _, row in per_model.iterrows():
        ax.annotate(
            row["model"],
            (row["billion_params"], row[y_col]),
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Billion parameters")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="Reasoning capacity")
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_size_vs_cost(df: pd.DataFrame, output_path: str, metric: str) -> None:
    """Parameter count vs generation cost, coloured by reasoning capacity."""
    _plot_model_size_scatter(
        df,
        output_path,
        metric,
        y_col="mean_time",
        y_label="Mean total generation time (s)",
        title="Model size vs generation cost",
    )


def plot_model_size_vs_quality(df: pd.DataFrame, output_path: str, metric: str) -> None:
    """Parameter count vs final quality, coloured by reasoning capacity."""
    _plot_model_size_scatter(
        df,
        output_path,
        metric,
        y_col="mean_metric",
        y_label=f"Mean final {metric}",
        title="Model size vs final quality",
    )


def main(results_csv: str, output_dir: str, metric: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(results_csv)

    plot_error_trajectory(
        df, os.path.join(output_dir, f"trajectory_{metric}.png"), metric
    )
    plot_final_distributions(
        df, os.path.join(output_dir, f"distributions_final_{metric}.png"), metric
    )
    plot_cost_quality(
        df, os.path.join(output_dir, f"cost_quality_{metric}.png"), metric
    )
    plot_model_size_vs_cost(
        df, os.path.join(output_dir, f"model_size_vs_cost_{metric}.png"), metric
    )
    plot_model_size_vs_quality(
        df, os.path.join(output_dir, f"model_size_vs_quality_{metric}.png"), metric
    )

    print(f"Plots written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument(
        "--metric", default="errors", choices=["errors", "warnings", "infos"]
    )
    args = parser.parse_args()
    main(args.results_csv, args.output_dir, args.metric)
