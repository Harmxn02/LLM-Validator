"""Statistical analysis of the batch experiment results.

Turns the raw per-run/per-iteration CSV into three things a course report
actually needs instead of eyeballed before/after deltas:

1. Per-model/condition mean +/- 95% CI for errors/warnings/infos on the final
   iteration of each run.
2. A paired Wilcoxon signed-rank test per model: does the feedback reprompt
   loop significantly reduce errors versus the initial generation? (paired on
   run_id, since it's the same generation being re-prompted.)
3. A paired Wilcoxon test per model comparing the feedback vs blind-ablation
   conditions' final error counts (paired because both conditions start from
   the same initial HTML for a given run_id) — isolates how much of any
   improvement is attributable to the validator feedback itself, versus just
   giving the model a second attempt.

Usage:
    uv run python -m analysis.stats --results-csv results/experiments.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.results_store import load_results

METRICS = ("errors", "warnings", "infos")


def mean_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
	"""Return (mean, ci_low, ci_high) using a t-distribution interval."""
	values = np.asarray(values, dtype=float)
	n = len(values)
	if n == 0:
		return float("nan"), float("nan"), float("nan")
	mean = float(values.mean())
	if n < 2 or values.std(ddof=1) == 0:
		return mean, mean, mean
	sem = scipy_stats.sem(values)
	lo, hi = scipy_stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
	return mean, float(lo), float(hi)


def _final_iteration_per_run(df: pd.DataFrame, condition: str) -> pd.DataFrame:
	"""Reprompt loops can stop early once errors hit 0, so the 'final' row per
	run is whichever iteration was logged last, not always the max configured
	iteration count."""
	subset = df[df["condition"] == condition]
	return subset.sort_values("iteration").groupby("run_id", as_index=False).tail(1)


def summarize_by_model_condition(df: pd.DataFrame) -> pd.DataFrame:
	finals = pd.concat(
		[_final_iteration_per_run(df, c) for c in ("feedback", "blind")]
		+ [df[df["condition"] == "initial"]]
	)
	rows = []
	for (model, condition), group in finals.groupby(["model", "condition"]):
		for metric in METRICS:
			mean, lo, hi = mean_ci(group[metric])
			rows.append(
				{
					"model": model,
					"condition": condition,
					"metric": metric,
					"n": len(group),
					"mean": round(mean, 3),
					"ci_low": round(lo, 3),
					"ci_high": round(hi, 3),
				}
			)
	return pd.DataFrame(rows)


def paired_before_after_test(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
	"""Does reprompting under `condition` significantly change `metric` vs the
	initial (pre-reprompt) generation, per model?"""
	initial = df[df["condition"] == "initial"][["run_id", "model", metric]].rename(
		columns={metric: "before"}
	)
	finals = _final_iteration_per_run(df, condition)[["run_id", metric]].rename(
		columns={metric: "after"}
	)
	merged = initial.merge(finals, on="run_id")

	results = []
	for model, group in merged.groupby("model"):
		before, after = group["before"].to_numpy(), group["after"].to_numpy()
		if len(before) < 2 or np.all(before == after):
			stat, p = float("nan"), float("nan")
		else:
			stat, p = scipy_stats.wilcoxon(before, after)
		results.append(
			{
				"model": model,
				"condition": condition,
				"metric": metric,
				"n": len(group),
				"mean_before": round(before.mean(), 3),
				"mean_after": round(after.mean(), 3),
				"wilcoxon_stat": stat,
				"p_value": p,
				"significant_at_0.05": bool(p < 0.05) if not np.isnan(p) else None,
			}
		)
	return pd.DataFrame(results)


def feedback_vs_blind_test(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	"""Paired comparison of feedback vs blind-ablation final error counts,
	paired on run_id since both conditions started from the same initial HTML."""
	feedback = _final_iteration_per_run(df, "feedback")[["run_id", "model", metric]]
	blind = _final_iteration_per_run(df, "blind")[["run_id", metric]].rename(
		columns={metric: "blind"}
	)
	merged = feedback.rename(columns={metric: "feedback"}).merge(blind, on="run_id")

	results = []
	for model, group in merged.groupby("model"):
		fb, bl = group["feedback"].to_numpy(), group["blind"].to_numpy()
		if len(fb) < 2 or np.all(fb == bl):
			stat, p = float("nan"), float("nan")
		else:
			stat, p = scipy_stats.wilcoxon(fb, bl)
		results.append(
			{
				"model": model,
				"metric": metric,
				"n": len(group),
				"mean_feedback": round(fb.mean(), 3),
				"mean_blind": round(bl.mean(), 3),
				"wilcoxon_stat": stat,
				"p_value": p,
				"significant_at_0.05": bool(p < 0.05) if not np.isnan(p) else None,
			}
		)
	return pd.DataFrame(results)


def top_error_categories(df: pd.DataFrame, condition: str = "initial", top_n: int = 10) -> pd.Series:
	"""Which error categories are most common in the given condition, across all runs."""
	counts: dict[str, int] = {}
	for categories in df[df["condition"] == condition]["error_categories"]:
		for cat, n in categories.items():
			counts[cat] = counts.get(cat, 0) + n
	return pd.Series(counts).sort_values(ascending=False).head(top_n)


def main(results_csv: str) -> None:
	df = load_results(results_csv)

	print("\n=== Mean +/- 95% CI by model/condition (final iteration per run) ===")
	print(summarize_by_model_condition(df).to_string(index=False))

	for condition in ("feedback", "blind"):
		print(f"\n=== Paired before/after test — {condition} reprompting ===")
		for metric in METRICS:
			result = paired_before_after_test(df, condition, metric)
			if not result.empty:
				print(result.to_string(index=False))

	print("\n=== Feedback vs blind ablation (paired, same initial HTML) ===")
	for metric in METRICS:
		result = feedback_vs_blind_test(df, metric)
		if not result.empty:
			print(result.to_string(index=False))

	print("\n=== Top error categories in initial generations ===")
	print(top_error_categories(df, "initial").to_string())


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--results-csv", default="results/experiments.csv", help="Path to the results CSV."
	)
	args = parser.parse_args()
	main(args.results_csv)
