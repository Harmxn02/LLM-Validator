"""Plots for the batch experiment results: convergence trajectories, per-model
spread, and a cost/quality tradeoff view.

Usage:
    uv run python -m analysis.plots --results-csv results/experiments.csv --output-dir plots
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import _final_iteration_per_run
from util.results_store import load_results


def _build_trajectory(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
	"""Per (model, run_id) series of `metric` across iterations, with iteration 0
	taken from the initial generation. Runs that stopped early (0 issues) have
	their last value carried forward, since that's the true count at later
	iterations — the loop just had nothing left to fix."""
	initial = df[df["condition"] == "initial"].set_index(["model", "run_id"])[metric]
	cond_df = df[df["condition"] == condition]
	pivot = cond_df.pivot_table(index=["model", "run_id"], columns="iteration", values=metric)
	pivot[0] = initial
	pivot = pivot.reindex(columns=sorted(pivot.columns))
	return pivot.ffill(axis=1)


def plot_error_trajectory(df: pd.DataFrame, output_path: str, condition: str, metric: str) -> None:
	pivot = _build_trajectory(df, condition, metric)

	fig, ax = plt.subplots(figsize=(8, 5))
	for model in sorted(pivot.index.get_level_values("model").unique()):
		sub = pivot.xs(model, level="model")
		means = sub.mean(axis=0)
		sems = sub.sem(axis=0)
		ax.errorbar(means.index, means.values, yerr=sems.values, marker="o", capsize=3, label=model)

	ax.set_xlabel("Reprompt iteration (0 = initial generation)")
	ax.set_ylabel(metric.capitalize())
	ax.set_title(f"{metric.capitalize()} vs iteration — {condition} condition")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_final_boxplot(df: pd.DataFrame, output_path: str, metric: str) -> None:
	models = sorted(df["model"].unique())
	conditions = ["initial", "feedback", "blind"]

	fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
	if len(models) == 1:
		axes = [axes]

	for ax, model in zip(axes, models):
		data = []
		for condition in conditions:
			if condition == "initial":
				sub = df[(df["model"] == model) & (df["condition"] == "initial")]
			else:
				sub = _final_iteration_per_run(df[df["model"] == model], condition)
			data.append(sub[metric].to_numpy())
		ax.boxplot(data, tick_labels=conditions)
		ax.set_title(model)
		ax.set_ylabel(metric.capitalize())

	fig.suptitle(f"Final {metric} by condition")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_cost_quality(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Mean total generation time (initial + feedback reprompts) per run vs mean
	final error count, one point per model — a faster model that self-corrects
	worse is a real tradeoff, not a strictly worse choice."""
	finals = _final_iteration_per_run(df, "feedback")[["run_id", "model", metric]]
	initial_time = df[df["condition"] == "initial"][["run_id", "gen_time_s"]].rename(
		columns={"gen_time_s": "initial_time"}
	)
	feedback_time = (
		df[df["condition"] == "feedback"].groupby("run_id")["gen_time_s"].sum().rename("feedback_time")
	)

	merged = finals.merge(initial_time, on="run_id").merge(feedback_time, on="run_id")
	merged["total_time_s"] = merged["initial_time"] + merged["feedback_time"]

	per_model = merged.groupby("model").agg(mean_time=("total_time_s", "mean"), mean_metric=(metric, "mean"))

	fig, ax = plt.subplots(figsize=(6, 5))
	ax.scatter(per_model["mean_time"], per_model["mean_metric"])
	for model, row in per_model.iterrows():
		ax.annotate(model, (row["mean_time"], row["mean_metric"]))
	ax.set_xlabel("Mean total generation time per run (s)")
	ax.set_ylabel(f"Mean final {metric} (feedback condition)")
	ax.set_title("Cost vs quality tradeoff")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def main(results_csv: str, output_dir: str, metric: str) -> None:
	os.makedirs(output_dir, exist_ok=True)
	df = load_results(results_csv)

	plot_error_trajectory(df, os.path.join(output_dir, f"trajectory_feedback_{metric}.png"), "feedback", metric)
	plot_error_trajectory(df, os.path.join(output_dir, f"trajectory_blind_{metric}.png"), "blind", metric)
	plot_final_boxplot(df, os.path.join(output_dir, f"boxplot_final_{metric}.png"), metric)
	plot_cost_quality(df, os.path.join(output_dir, f"cost_quality_{metric}.png"), metric)

	print(f"Plots written to {output_dir}/")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results-csv", default="results/experiments.csv")
	parser.add_argument("--output-dir", default="plots")
	parser.add_argument("--metric", default="errors", choices=["errors", "warnings", "infos"])
	args = parser.parse_args()
	main(args.results_csv, args.output_dir, args.metric)
