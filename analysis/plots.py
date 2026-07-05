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
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import _final_iteration_per_run, build_trajectory
from util.results_store import load_results

# Static metadata about each local model, since it isn't derivable from the
# results CSV. Feed this into plot_model_characteristics(). Sizes are billions
# of parameters and on-disk GB; context is in thousands of tokens.
MODEL_INFO = {
	"gemma4:e2b": {"reasoning": True, "billion_params": 5.12},
	"qwen3:8b": {"reasoning": True, "billion_params": 8.19},
	"qwen2.5-coder:14b": {"reasoning": False, "billion_params": 14.8},
	"qwen3.5:9b": {"reasoning": True, "billion_params": 9.65},
}


def plot_error_trajectory(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Mean metric vs reprompt iteration, one coloured line per model, with the
	feedback and blind conditions shown as side-by-side panels sharing a
	y-axis so the two correction strategies are directly comparable. Shaded
	SEM bands replace errorbars for a calmer read when lines overlap."""
	conditions = ["feedback", "blind"]
	models = sorted(df["model"].unique())
	colours = dict(zip(models, sns.color_palette("tab10", n_colors=len(models))))

	fig, axes = plt.subplots(1, len(conditions), figsize=(11, 5), sharey=True)

	for ax, condition in zip(axes, conditions):
		pivot = build_trajectory(df, condition, metric)
		for model in models:
			sub = pivot.xs(model, level="model")
			means = sub.mean(axis=0)
			sems = sub.sem(axis=0)
			ax.plot(
				means.index, means.values, marker="o", label=model, color=colours[model]
			)
			ax.fill_between(
				means.index,
				means.values - sems.values,
				means.values + sems.values,
				color=colours[model],
				alpha=0.2,
			)
		ax.set_xlabel("Reprompt iteration (0 = initial generation)")
		ax.set_title(f"{condition.capitalize()} condition")

	axes[0].set_ylabel(metric.capitalize())
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		loc="upper center",
		ncol=len(labels),
		bbox_to_anchor=(0.5, 1.06),
	)
	fig.suptitle(f"{metric.capitalize()} vs reprompt iteration", y=1.1)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def plot_final_distributions(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Final metric distribution per model, grouped by condition, using a
	violin plus jittered strip overlay on one set of axes rather than one
	boxplot subplot per model — every model and condition sits on the same
	scale so the spread is directly comparable at a glance."""
	conditions = ["initial", "feedback", "blind"]
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

	fig, ax = plt.subplots(figsize=(8, 5))
	sns.violinplot(
		data=long_df,
		x="model",
		y=metric,
		hue="condition",
		ax=ax,
		cut=0,
		inner=None,
		alpha=0.6,
	)
	sns.stripplot(
		data=long_df,
		x="model",
		y=metric,
		hue="condition",
		ax=ax,
		dodge=True,
		size=4,
		color="black",
		alpha=0.5,
		legend=False,
	)
	ax.set_xlabel("Model")
	ax.set_ylabel(f"Final {metric.capitalize()}")
	ax.set_title(f"Final {metric} distribution by model and condition")
	ax.legend(title="Condition")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def _per_model_cost_quality(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	"""Mean total generation time (initial + feedback reprompts) per run and
	mean final metric, aggregated to one row per model. Shared by the
	cost/quality plot and the model-characteristics plot so both read from a
	single source of truth."""
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
	self-corrects worse is a real tradeoff, not a strictly worse choice. A red
	linear regression line (via Seaborn) is overlaid to highlight the overall
	trend."""
	per_model = _per_model_cost_quality(df, metric)

	fig, ax = plt.subplots(figsize=(6, 5))
	sns.regplot(
		data=per_model,
		x="mean_time",
		y="mean_metric",
		ax=ax,
		scatter_kws={"s": 50, "color": "steelblue", "zorder": 2},
		line_kws={"color": "red", "linewidth": 2},
		ci=None,
	)
	for _, row in per_model.iterrows():
		ax.annotate(row["model"], (row["mean_time"], row["mean_metric"]))
	ax.set_xlabel("Mean total generation time per run (s)")
	ax.set_ylabel(f"Mean final {metric} (feedback condition)")
	ax.set_title("Cost vs quality tradeoff")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def plot_model_characteristics(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Parameter count vs cost and vs quality, coloured by whether the model
	has reasoning capacity. Two panels share the parameter-count x-axis so
	cost and quality can be read off side-by-side for the same model. Models
	missing from MODEL_INFO are silently dropped."""
	per_model = _per_model_cost_quality(df, metric)
	info = pd.DataFrame.from_dict(MODEL_INFO, orient="index").reset_index(names="model")
	per_model = per_model.merge(info, on="model", how="inner")
	per_model["reasoning_label"] = per_model["reasoning"].map(
		{True: "Reasoning", False: "No reasoning"}
	)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	panels = [
		("mean_time", "Mean total generation time (s)"),
		("mean_metric", f"Mean final {metric}"),
	]

	for ax, (y_col, y_label) in zip(axes, panels):
		sns.scatterplot(
			data=per_model,
			x="billion_params",
			y=y_col,
			hue="reasoning_label",
			style="reasoning_label",
			s=150,
			ax=ax,
		)
		for _, row in per_model.iterrows():
			ax.annotate(
				row["model"],
				(row["billion_params"], row[y_col]),
				fontsize=8,
				xytext=(6, 6),
				textcoords="offset points",
			)
		ax.set_xlabel("Billion parameters")
		ax.set_ylabel(y_label)
		ax.legend(title="Reasoning capacity")

	fig.suptitle("Model size and reasoning capacity vs cost and quality")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


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
	plot_model_characteristics(
		df, os.path.join(output_dir, f"model_characteristics_{metric}.png"), metric
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
