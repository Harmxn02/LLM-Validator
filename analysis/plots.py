"""Plots for the batch experiment results: convergence trajectories, per-model
spread, and cost/quality/characteristics views.

Usage:
	uv run python -m analysis.plots --results-csv results/experiments.csv --output-dir plots
"""

import argparse
import os
import sys

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import (
	_final_iteration_per_run,
	build_trajectory,
	summarize_by_model_condition,
)
from util.results_store import load_results

MODEL_INFO_PATH = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	"config",
	"model_info.yaml",
)

# One shared theme for every figure in this module, so a reader flipping
# between plots sees one consistent visual system rather than a new style
# per chart. palette=None keeps seaborn's own default ("deep") categorical
# palette rather than pinning to a specific named one, so every categorical
# colour mapping below (model identity, thinking mode, condition) is drawn
# from that same single default cycle.
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update(
	{
		"figure.dpi": 300,
		"savefig.dpi": 300,
		"font.family": "sans-serif",
		"font.sans-serif": [
			"Helvetica Neue",
			"Helvetica",
			"Arial",
			"Nimbus Sans",
			"Liberation Sans",
			"DejaVu Sans",
		],
		"axes.titleweight": "bold",
		"axes.titlesize": 13,
		"axes.labelsize": 11,
		"axes.edgecolor": "0.3",
		"legend.frameon": True,
		"legend.framealpha": 0.9,
		"legend.edgecolor": "0.85",
		"legend.fancybox": True,
		"legend.fontsize": 9.5,
		"legend.title_fontsize": 10,
	}
)

DEFAULT_PALETTE = sns.color_palette()

# Parameter count is shown as a binned category rather than a continuous size
# so that models of essentially the same size (e.g. 8.0B and 8.19B) render as
# the same dot size instead of two arbitrarily-different ones.
PARAM_BIN_ORDER = ["<1B", "1-5B", "6-10B", ">10B"]
PARAM_BIN_SIZES = {"<1B": 33, "1-5B": 100, "6-10B": 200, ">10B": 400}

CONDITION_ORDER = ["initial", "feedback", "blind"]

THINKING_LABELS = {True: "Thinking mode", False: "No thinking mode"}


def _load_model_info(path: str = MODEL_INFO_PATH) -> dict:
	"""Static per-model metadata that isn't derivable from the results CSV
	(config/model_info.yaml): billion_params (total, on-disk parameter count)
	and native_thinking_mode.

	native_thinking_mode means the model ships with a documented, toggleable
	extended-thinking / chain-of-thought mode (e.g. Qwen3's hybrid thinking
	mode) — a specific, checkable architectural feature, unlike the vaguer
	"does this model reason" framing. Confirmed for qwen3:8b, qwen2.5-coder:14b,
	qwen3.5:9b, and gemma4:e2b against this project's own testing notes
	(Models.md); gemma4:e4b is inferred from e2b by architecture family
	(Gemma's MatFormer e2b/e4b variants share training); gemma3:1b and
	smollm2:135m are unverified best guesses. Models missing from this file
	are silently dropped from the plots that need it.
	"""
	with open(path) as f:
		return yaml.safe_load(f)


MODEL_INFO = _load_model_info()


def _param_bin(billion_params: float) -> str:
	if billion_params < 1:
		return "<1B"
	if billion_params <= 5:
		return "1-5B"
	if billion_params <= 10:
		return "6-10B"
	return ">10B"


def _model_palette(models: list[str]) -> dict:
	"""One stable colour per model, drawn from the shared default palette, so
	the same model always reads as the same colour across every plot that
	breaks results down by model."""
	models = sorted(models)
	return dict(zip(models, sns.color_palette(n_colors=len(models))))


def _with_model_info(per_model: pd.DataFrame) -> pd.DataFrame:
	info = pd.DataFrame.from_dict(MODEL_INFO, orient="index").reset_index(names="model")
	merged = per_model.merge(info, on="model", how="inner")
	merged["thinking_label"] = merged["native_thinking_mode"].map(THINKING_LABELS)
	merged["param_bin"] = pd.Categorical(
		merged["billion_params"].map(_param_bin),
		categories=PARAM_BIN_ORDER,
		ordered=True,
	)
	return merged


def _thinking_palette() -> dict:
	# Index 0 and 3 of the default palette are its blue and red.
	return {
		THINKING_LABELS[True]: DEFAULT_PALETTE[0],
		THINKING_LABELS[False]: DEFAULT_PALETTE[3],
	}


def _strength(r2: float) -> str:
	if r2 >= 0.7:
		return "strongly"
	if r2 >= 0.3:
		return "moderately"
	return "weakly"


def _add_trend_line(ax, x: pd.Series, y: pd.Series) -> dict | None:
	"""Grey dashed least-squares fit, shared by every scatterplot in this
	module. Skipped when there aren't at least 2 distinct x values, since a
	line through one point (or a vertical stack of points) isn't meaningful.
	Returns the slope and R² so callers can report it in the subtitle
	instead — no on-plot text box, to avoid saying R² twice."""
	if len(x) < 2 or x.nunique() < 2:
		return None
	coeffs = np.polyfit(x, y, 1)
	r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
	xs = np.linspace(x.min(), x.max(), 50)
	ax.plot(
		xs,
		np.polyval(coeffs, xs),
		color="0.55",
		linestyle="--",
		linewidth=1.5,
		zorder=1,
	)
	return {"slope": float(coeffs[0]), "r2": r2}


# Section-header rows that seaborn injects into a combined hue+size legend —
# bolding these (instead of leaving every row the same weight) is what turns
# "a flat list of words" into a visually structured, two-level legend.
_LEGEND_SECTION_TITLES = {"Native thinking mode", "Parameters", "Condition"}


def _style_legend(leg) -> None:
	if leg is None:
		return
	title = leg.get_title()
	if title.get_text():
		title.set_fontweight("bold")
	for text in leg.get_texts():
		if text.get_text() in _LEGEND_SECTION_TITLES:
			text.set_fontweight("bold")


def _titled(ax, title: str, subtitle: str | None) -> None:
	"""Bold title plus an italic, muted-grey subtitle carrying the plot's
	key finding, stacked directly beneath it. Positions are in points (not
	axes-fraction) so the subtitle sits snug under the title with a small,
	fixed gap above the plot itself rather than drifting with figure size."""
	ax.set_title(title, fontsize=13, fontweight="bold", pad=34 if subtitle else 10)
	if subtitle:
		ax.annotate(
			subtitle,
			xy=(0.5, 1.0),
			xycoords="axes fraction",
			xytext=(0, 14),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=9.5,
			style="italic",
			color="0.35",
			annotation_clip=False,
		)


def _safe_pct_diff(a: float, b: float) -> float:
	"""Percent by which a is smaller than b, guarding the b == 0 case."""
	return (b - a) / b * 100 if b else 0.0


def plot_error_trajectory(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Mean metric vs reprompt iteration, one coloured line per model, with the
	feedback and blind conditions shown as side-by-side panels sharing a
	y-axis so the two correction strategies are directly comparable. Shaded
	SEM bands replace errorbars for a calmer read when lines overlap."""
	df = df[df["model"].isin(MODEL_INFO)]
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
		bbox_to_anchor=(0.5, 1.05),
	)

	fb_mean = _final_iteration_per_run(df, "feedback")[metric].mean()
	bl_mean = _final_iteration_per_run(df, "blind")[metric].mean()
	if fb_mean < bl_mean:
		subtitle = f"Feedback reprompting ends {_safe_pct_diff(fb_mean, bl_mean):.0f}% lower on average than blind reprompting."
	elif bl_mean < fb_mean:
		subtitle = f"Blind reprompting ends {_safe_pct_diff(bl_mean, fb_mean):.0f}% lower on average than feedback reprompting."
	else:
		subtitle = (
			f"Feedback and blind reprompting end with the same average final {metric}."
		)

	fig.suptitle(
		f"{metric.capitalize()} vs reprompt iteration",
		y=1.22,
		fontsize=15,
		fontweight="bold",
	)
	fig.text(
		0.5, 1.13, subtitle, ha="center", fontsize=10, style="italic", color="0.35"
	)
	sns.despine(fig)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def plot_final_summary_bars(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Mean final metric (95% CI) per model, grouped by condition, as a bar
	chart rather than a per-model box/strip plot. Bars scale to any number of
	models by widening the figure and rotating labels, whereas a box+strip
	layout runs out of room per model well before that; the full per-run
	distribution this collapses is better suited to a table than a plot
	anyway once there are many models."""
	df = df[df["model"].isin(MODEL_INFO)]
	summary = summarize_by_model_condition(df)
	summary = summary[summary["metric"] == metric]
	models = sorted(summary["model"].unique())
	conditions = [c for c in CONDITION_ORDER if c in summary["condition"].unique()]
	palette = dict(zip(conditions, sns.color_palette(n_colors=len(conditions))))

	fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(models) + 3), 5.5))
	x = np.arange(len(models))
	width = 0.8 / max(len(conditions), 1)

	for i, condition in enumerate(conditions):
		sub = (
			summary[summary["condition"] == condition]
			.set_index("model")
			.reindex(models)
		)
		means = sub["mean"].to_numpy()
		lo = np.clip(means - sub["ci_low"].to_numpy(), 0, None)
		hi = np.clip(sub["ci_high"].to_numpy() - means, 0, None)
		offsets = x + (i - (len(conditions) - 1) / 2) * width
		ax.bar(
			offsets,
			means,
			width=width * 0.9,
			label=condition.capitalize(),
			color=palette[condition],
			yerr=[lo, hi],
			capsize=3,
			edgecolor="white",
			linewidth=0.6,
		)

	ax.set_xticks(x)
	ax.set_xticklabels(models, rotation=0, ha="right")
	ax.set_ylabel(f"Mean final {metric} (95% CI)")
	ax.grid(alpha=0.3, axis="y")
	_style_legend(ax.legend(title="Condition"))

	if not summary.empty:
		best = summary.loc[summary["mean"].idxmin()]
		subtitle = (
			f"Lowest mean final {metric}: {best['model']} under the "
			f"{best['condition']} condition ({best['mean']:.2f})."
		)
	else:
		subtitle = None
	_titled(ax, f"Final {metric} by model and condition", subtitle)

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
	Colour encodes native thinking mode and point size encodes a binned
	parameter-count range, so speed, quality, and size all show up in one
	figure."""
	per_model = _with_model_info(_per_model_cost_quality(df, metric))
	plot_df = per_model.rename(
		columns={"thinking_label": "Native thinking mode", "param_bin": "Parameters"}
	)

	fig, ax = plt.subplots(figsize=(7.5, 5.5))
	trend = _add_trend_line(ax, per_model["mean_time"], per_model["mean_metric"])

	sns.scatterplot(
		data=plot_df,
		x="mean_time",
		y="mean_metric",
		hue="Native thinking mode",
		palette=_thinking_palette(),
		size="Parameters",
		sizes=PARAM_BIN_SIZES,
		size_order=PARAM_BIN_ORDER,
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
	ax.grid(alpha=0.3)
	_style_legend(
		ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
	)

	if trend:
		direction = "fewer" if trend["slope"] < 0 else "more"
		subtitle = (
			f"Longer generation time is {_strength(trend['r2'])} associated with "
			f"{direction} final {metric} (R² = {trend['r2']:.2f})."
		)
	else:
		subtitle = None
	_titled(ax, "Cost vs quality tradeoff", subtitle)

	sns.despine(fig)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def _plot_model_size_scatter(
	df: pd.DataFrame,
	output_path: str,
	metric: str,
	y_col: str,
	y_label: str,
	title: str,
	subtitle_fn,
) -> None:
	per_model = _with_model_info(_per_model_cost_quality(df, metric))
	plot_df = per_model.rename(columns={"thinking_label": "Native thinking mode"})

	fig, ax = plt.subplots(figsize=(7, 5.5))
	trend = _add_trend_line(ax, per_model["billion_params"], per_model[y_col])

	sns.scatterplot(
		data=plot_df,
		x="billion_params",
		y=y_col,
		hue="Native thinking mode",
		style="Native thinking mode",
		palette=_thinking_palette(),
		s=180,
		edgecolor="white",
		linewidth=0.8,
		ax=ax,
		zorder=2,
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
	ax.grid(alpha=0.3)
	_style_legend(ax.legend(title="Native thinking mode"))
	_titled(ax, title, subtitle_fn(trend) if trend else None)

	sns.despine(fig)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def plot_model_size_vs_cost(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Parameter count vs generation cost, coloured by native thinking mode."""
	_plot_model_size_scatter(
		df,
		output_path,
		metric,
		y_col="mean_time",
		y_label="Mean total generation time (s)",
		title="Model size vs generation cost",
		subtitle_fn=lambda t: (
			f"Generation cost scales {_strength(t['r2'])} with parameter count (R² = {t['r2']:.2f})."
		),
	)


def plot_model_size_vs_quality(df: pd.DataFrame, output_path: str, metric: str) -> None:
	"""Parameter count vs final quality, coloured by native thinking mode."""
	_plot_model_size_scatter(
		df,
		output_path,
		metric,
		y_col="mean_metric",
		y_label=f"Mean final {metric}",
		title="Model size vs final quality",
		subtitle_fn=lambda t: (
			f"Larger models tend to produce {'fewer' if t['slope'] < 0 else 'more'} "
			f"final {metric}, {_strength(t['r2'])} (R² = {t['r2']:.2f})."
		),
	)


def main(results_csv: str, output_dir: str, metric: str) -> None:
	os.makedirs(output_dir, exist_ok=True)
	df = load_results(results_csv)

	plot_error_trajectory(
		df, os.path.join(output_dir, f"trajectory_{metric}.png"), metric
	)
	plot_final_summary_bars(
		df, os.path.join(output_dir, f"final_summary_{metric}.png"), metric
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
