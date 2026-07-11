"""Generate copy-paste-ready LaTeX tables for the report, from the same
analyses as stats.py. Each table is emitted as a self-contained
`table` environment with its own \\caption/\\label, preceded by a comment
noting which report section/paragraph it belongs to.

Usage:
	uv run python -m analysis.tables --results-csv results/experiments.csv --output tables.tex
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stats import (
	DIFFICULTY_ORDER,
	aggregate_before_after_test,
	error_category_before_after,
	error_reduction_by_difficulty_reasoning,
	iteration_cutoff_analysis,
	token_cost_comparison,
)
from util.model_info import load_model_info
from util.results_store import load_results


def _tex_escape(s: str) -> str:
	"""Escape the handful of LaTeX-special characters that could plausibly
	show up in a model name or category label (none currently do, but a
	stray underscore in a future model name shouldn't silently break the
	build)."""
	return re.sub(r"([_%&#])", r"\\\1", str(s))


def _fmt_p(p: float | None) -> str:
	if p is None or pd.isna(p):
		return "---"
	return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def _fmt_signed(x: float, decimals: int = 1) -> str:
	return f"{x:+.{decimals}f}"


def model_summary_table(model_info: dict) -> str:
	"""Methodology \\S Models & Configurations, paragraph 3: model name,
	parameter count, reasoning yes/no, quantisation (always no here)."""
	rows = []
	for model in sorted(model_info, key=lambda m: model_info[m]["billion_params"]):
		info = model_info[model]
		reasoning = "Yes" if info["native_thinking_mode"] else "No"
		rows.append(
			f"{_tex_escape(model)} & {info['billion_params']:g} & {reasoning} & No \\\\"
		)
	body = "\n".join(rows)
	return f"""% === Methodology \\S Models & Configurations (paragraph 3: model summary table) ===
\\begin{{table}}[H]
\\centering
\\caption{{Models evaluated, spanning a range of parameter counts and both reasoning and non-reasoning architectures. All models run locally via Ollama at full precision (no quantisation). smollm2:135m was also run as a sanity-check baseline but is excluded here and from every model-info-dependent analysis, since it has no confirmed reasoning-capability entry.}}
\\label{{tab:model-summary}}
\\begin{{tabular}}{{lccc}}
\\toprule
Model & Parameters (B) & Reasoning & Quantised \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def aggregate_summary_table(df: pd.DataFrame) -> str:
	"""RQ1 paragraph 2: aggregate before/after error and warning counts
	across all models and all 51 prompts, feedback condition, with the
	paired Wilcoxon test and 95% CI on the reduction."""
	rows = []
	n = None
	for metric in ("errors", "warnings"):
		r = aggregate_before_after_test(df, "feedback", metric)
		n = r["n"]
		rows.append(
			f"{metric.capitalize()} & {r['mean_before']:.2f} & {r['mean_after']:.2f} & "
			f"{r['delta']:.2f} & {r['pct_reduction']:.1f}\\% & {_fmt_p(r['p_value'])} & "
			f"[{r['reduction_ci_low']:.2f}, {r['reduction_ci_high']:.2f}] \\\\"
		)
	body = "\n".join(rows)
	return f"""% === RQ1 paragraph 2: aggregate before/after summary (feedback condition) ===
\\begin{{table}}[H]
\\centering
\\caption{{Aggregate before/after comparison across all models and all 51 prompts under the feedback condition ($n={n}$ runs). The 95\\% CI is on the paired per-run reduction (before $-$ after); $p$-values are from a paired Wilcoxon signed-rank test against the initial (pre-reprompt) generation.}}
\\label{{tab:rq1-aggregate}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
Metric & Mean before & Mean after & $\\Delta$ & \\% reduction & $p$ & 95\\% CI ($\\Delta$) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def feedback_vs_blind_reasoning_table(df: pd.DataFrame) -> str:
	"""RQ1 paragraph 3: feedback vs blind mean error reduction, split by
	difficulty tier (rows) and reasoning/non-reasoning model category
	(columns) — 2 columns x 6 rows as sketched in the report outline."""
	result = error_reduction_by_difficulty_reasoning(df, "errors")
	value_pivot = result.pivot(
		index=["difficulty", "condition"], columns="reasoning_category",
		values="mean_reduction",
	)
	n_pivot = result.pivot(
		index=["difficulty", "condition"], columns="reasoning_category", values="n"
	)

	rows = []
	for difficulty in DIFFICULTY_ORDER:
		for condition in ("feedback", "blind"):
			key = (difficulty, condition)
			if key not in value_pivot.index:
				continue
			reasoning_val = value_pivot.loc[key, "Reasoning"]
			nonreasoning_val = value_pivot.loc[key, "Non-reasoning"]
			reasoning_n = int(n_pivot.loc[key, "Reasoning"])
			nonreasoning_n = int(n_pivot.loc[key, "Non-reasoning"])
			label = f"{difficulty.capitalize()} -- {condition.capitalize()}"
			rows.append(
				f"{label} & {reasoning_val:.2f} (n={reasoning_n}) & "
				f"{nonreasoning_val:.2f} (n={nonreasoning_n}) \\\\"
			)
	body = "\n".join(rows)
	return f"""% === RQ1 paragraph 3: feedback vs blind ablation, by difficulty x reasoning category ===
\\begin{{table}}[H]
\\centering
\\caption{{Mean error reduction (before $-$ after) by prompt difficulty tier and condition, split by whether the model has a native reasoning/thinking mode. Higher values mean more errors were resolved. If the feedback condition's reduction consistently exceeds blind's within each row, that isolates the validator's specific error list as doing real work, beyond the mere act of reprompting.}}
\\label{{tab:rq1-feedback-vs-blind-reasoning}}
\\begin{{tabular}}{{lcc}}
\\toprule
Difficulty -- condition & Reasoning models & Non-reasoning models \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def error_category_table(df: pd.DataFrame, top_n: int = 10) -> str:
	"""RQ1 paragraph 5: which error categories the guardrail is most/least
	effective at resolving, before vs after feedback reprompting."""
	result = error_category_before_after(df, "feedback", top_n=top_n)
	rows = []
	for _, r in result.iterrows():
		pct = "---" if pd.isna(r["pct_resolved"]) else f"{r['pct_resolved']:.1f}\\%"
		rows.append(
			f"{_tex_escape(r['category'])} & {int(r['before'])} & {int(r['after'])} & "
			f"{int(r['delta'])} & {pct} \\\\"
		)
	body = "\n".join(rows)
	return f"""% === RQ1 paragraph 5: error category breakdown, before vs after feedback reprompting ===
\\begin{{table}}[H]
\\centering
\\caption{{Frequency of each W3C validator message category before (initial generation) and after (final feedback-condition iteration), across all models and runs. Categories are drawn from \\texttt{{util.validation.categorize\\_error}} and combine both error- and warning-level messages; ``other'' is the uncategorised catch-all. Sorted by pre-reprompt frequency.}}
\\label{{tab:rq1-error-categories}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Category & Before & After & $\\Delta$ & \\% resolved \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def token_cost_table(df: pd.DataFrame) -> str:
	"""RQ2 paragraph 4: does the feedback condition cost more tokens than
	blind, and is the extra cost (if any) justified by the quality gain?"""
	result = token_cost_comparison(df, "errors")
	rows = []
	for _, r in result.iterrows():
		label = (
			"\\textbf{All models (pooled)}"
			if r["model"] == "ALL"
			else _tex_escape(r["model"])
		)
		ratio = (
			"---"
			if r["tokens_per_extra_resolved"] is None
			or pd.isna(r["tokens_per_extra_resolved"])
			else f"{r['tokens_per_extra_resolved']:.1f}"
		)
		rows.append(
			f"{label} & {r['mean_feedback_tokens']:.0f} & {r['mean_blind_tokens']:.0f} & "
			f"{_fmt_signed(r['extra_tokens'], 0)} ({_fmt_signed(r['extra_tokens_pct'])}\\%) & "
			f"{_fmt_signed(r['extra_resolved'], 3)} & {ratio} \\\\"
		)
	body = "\n".join(rows)
	return f"""% === RQ2 paragraph 4: token cost, feedback vs blind ===
% Wrapped in \\resizebox since this table is wide (6 columns) -- confirmed it
% overflows a single-column page width otherwise (tested standalone).
\\begin{{table}}[H]
\\centering
\\caption{{Mean total tokens per run (initial generation plus all reprompt iterations) under feedback vs blind, per model. ``Extra tokens'' is feedback minus blind; ``extra resolved'' is the mean additional errors resolved by feedback over blind (positive = feedback resolves more). The last column is tokens spent per additional error resolved, only defined where feedback resolves strictly more errors than blind.}}
\\label{{tab:rq2-token-cost}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lccccc}}
\\toprule
Model & Mean feedback tokens & Mean blind tokens & Extra tokens (feedback $-$ blind) & Extra errors resolved & Tokens / extra resolved \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""


def iteration_cutoff_table(df: pd.DataFrame) -> str:
	"""Iteration Count Analysis paragraph 2: diminishing returns and the
	optimal iteration cutoff, broken down by difficulty tier."""
	result = iteration_cutoff_analysis(
		df, condition="feedback", metric="errors", group_by="difficulty"
	)
	rows = []
	for _, r in result.iterrows():
		rows.append(
			f"{r['difficulty'].capitalize()} & {int(r['iteration_cutoff'])} & "
			f"{r['pct_resolved']:.1f}\\% & {r['mean_remaining']:.2f} & {int(r['n_runs'])} \\\\"
		)
	body = "\n".join(rows)
	return f"""% === Iteration Count Analysis paragraph 2: cutoff analysis by difficulty tier (feedback condition) ===
\\begin{{table}}[H]
\\centering
\\caption{{For each iteration cap, the percentage of feedback-condition runs already fully resolved (zero errors) and the mean remaining error count at that cutoff, by prompt difficulty tier. The per-model equivalent is available via \\texttt{{analysis.stats.iteration\\_cutoff\\_analysis(..., group\\_by="model")}}.}}
\\label{{tab:iteration-cutoff-difficulty}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Difficulty & Iteration cutoff & \\% resolved & Mean remaining & $n$ runs \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main(results_csv: str, output_path: str) -> None:
	df = load_results(results_csv)
	model_info = load_model_info()

	sections = [
		model_summary_table(model_info),
		aggregate_summary_table(df),
		feedback_vs_blind_reasoning_table(df),
		error_category_table(df),
		token_cost_table(df),
		iteration_cutoff_table(df),
	]

	header = (
		"% Auto-generated by `uv run python -m analysis.tables`. Regenerate after any\n"
		"% change to results/experiments.csv rather than hand-editing the numbers below.\n"
		"% Copy individual \\begin{table}...\\end{table} blocks into the relevant report\n"
		"% section as needed -- this file is not \\input by main.tex.\n"
		"% Requires: booktabs, float, graphicx (all already loaded in report/main.tex).\n\n"
	)
	with open(output_path, "w") as f:
		f.write(header + "\n\n".join(sections))

	print(f"Tables written to {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results-csv", default="results/experiments.csv")
	parser.add_argument("--output", default="tables.tex")
	args = parser.parse_args()
	main(args.results_csv, args.output)
