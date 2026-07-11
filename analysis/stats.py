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

from util.model_info import load_model_info, reasoning_category
from util.results_store import load_results

METRICS = ("errors", "warnings", "infos")
DIFFICULTY_ORDER = ("simple", "medium", "difficult")


def _pct_reduction(after: float, before: float) -> float:
    """Percent by which `after` is smaller than `before`, guarding before == 0."""
    return (before - after) / before * 100 if before else 0.0


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


def build_trajectory(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    """Per (model, run_id) series of `metric` across iterations, with iteration 0
    taken from the initial generation. Runs that stopped early (0 issues) have
    their last value carried forward, since that's the true count at later
    iterations — the loop just had nothing left to fix. Shared by plots.py and
    the iteration-cutoff analysis below."""
    initial = df[df["condition"] == "initial"].set_index(["model", "run_id"])[metric]
    cond_df = df[df["condition"] == condition]
    pivot = cond_df.pivot_table(
        index=["model", "run_id"], columns="iteration", values=metric
    )
    pivot[0] = initial
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    return pivot.ffill(axis=1)


def iteration_cutoff_analysis(
    df: pd.DataFrame,
    condition: str = "feedback",
    metric: str = "errors",
    group_by: str = "model",
) -> pd.DataFrame:
    """For each possible iteration cap N (0..max configured iterations), report
    the fraction of runs already resolved (metric == 0) and the mean remaining
    value at that cutoff, grouped by `group_by` ('model' or 'difficulty').

    Directly answers "how many reprompt iterations are actually worth it" from
    data already collected by a single run_batch sweep — no need to re-run the
    experiment at different max-iteration configs. Grouping by difficulty
    (instead of model) answers whether the optimal cutoff differs by prompt
    difficulty rather than just by model.
    """
    pivot = build_trajectory(df, condition, metric)
    max_iter = int(pivot.columns.max())

    if group_by == "model":
        keys = sorted(pivot.index.get_level_values("model").unique())

        def _group(key: str) -> pd.DataFrame:
            return pivot.xs(key, level="model")
    elif group_by == "difficulty":
        difficulty_of_run = df.drop_duplicates("run_id").set_index("run_id")[
            "difficulty"
        ]
        run_difficulties = pivot.index.get_level_values("run_id").map(
            difficulty_of_run
        )
        present = set(run_difficulties)
        keys = [d for d in DIFFICULTY_ORDER if d in present]

        def _group(key: str) -> pd.DataFrame:
            return pivot[run_difficulties == key]
    else:
        raise ValueError(f"group_by must be 'model' or 'difficulty', got {group_by!r}")

    rows = []
    for key in keys:
        sub = _group(key)
        for cutoff in range(0, max_iter + 1):
            if cutoff not in sub.columns:
                continue
            values_at_cutoff = sub[cutoff]
            rows.append(
                {
                    group_by: key,
                    "condition": condition,
                    "iteration_cutoff": cutoff,
                    "n_runs": len(values_at_cutoff),
                    "pct_resolved": round((values_at_cutoff == 0).mean() * 100, 1),
                    "mean_remaining": round(values_at_cutoff.mean(), 3),
                }
            )
    return pd.DataFrame(rows)


def iterations_to_convergence(
    df: pd.DataFrame, condition: str = "feedback", metric: str = "errors"
) -> pd.DataFrame:
    """First iteration at which `metric` hits 0 for each (model, run_id) in
    `condition`, or NaN if the run never reaches 0 within the observed
    iterations. Feeds the convergence-distribution histogram — a direct,
    per-run answer to "how many reprompts does it actually take", rather than
    the mean-per-iteration view `build_trajectory` gives."""
    pivot = build_trajectory(df, condition, metric)

    def _first_zero(row: pd.Series) -> float:
        resolved = row.index[row == 0]
        return float(resolved.min()) if len(resolved) else float("nan")

    converged_at = pivot.apply(_first_zero, axis=1)
    result = converged_at.rename("iterations_to_convergence").reset_index()
    result["condition"] = condition
    return result


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


def paired_before_after_test(
    df: pd.DataFrame, condition: str, metric: str
) -> pd.DataFrame:
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


def aggregate_before_after_test(
    df: pd.DataFrame, condition: str, metric: str
) -> dict:
    """Same paired before/after test as `paired_before_after_test`, but pooled
    across every model rather than broken down per model — the aggregate
    headline number for "does the guardrail reduce errors overall", with a
    95% CI on the paired per-run reduction (before - after) rather than on
    the raw before/after means."""
    initial = df[df["condition"] == "initial"][["run_id", metric]].rename(
        columns={metric: "before"}
    )
    finals = _final_iteration_per_run(df, condition)[["run_id", metric]].rename(
        columns={metric: "after"}
    )
    merged = initial.merge(finals, on="run_id")
    before, after = merged["before"].to_numpy(), merged["after"].to_numpy()

    if len(before) < 2 or np.all(before == after):
        stat, p = float("nan"), float("nan")
    else:
        stat, p = scipy_stats.wilcoxon(before, after)

    reduction_mean, reduction_lo, reduction_hi = mean_ci(before - after)
    mean_before, mean_after = float(before.mean()), float(after.mean())
    return {
        "condition": condition,
        "metric": metric,
        "n": len(merged),
        "mean_before": round(mean_before, 3),
        "mean_after": round(mean_after, 3),
        "delta": round(mean_before - mean_after, 3),
        "pct_reduction": round(_pct_reduction(mean_after, mean_before), 1),
        "reduction_ci_low": round(reduction_lo, 3),
        "reduction_ci_high": round(reduction_hi, 3),
        "wilcoxon_stat": stat,
        "p_value": p,
        "significant_at_0.05": bool(p < 0.05) if not np.isnan(p) else None,
    }


def summarize_by_difficulty_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- 95% CI per difficulty tier/condition (final iteration per
    run), mirroring `summarize_by_model_condition` but grouped by prompt
    difficulty instead of model — feeds the RQ1 "does harder mean worse"
    breakdown."""
    finals = pd.concat(
        [_final_iteration_per_run(df, c) for c in ("feedback", "blind")]
        + [df[df["condition"] == "initial"]]
    )
    rows = []
    for (difficulty, condition), group in finals.groupby(["difficulty", "condition"]):
        for metric in METRICS:
            mean, lo, hi = mean_ci(group[metric])
            rows.append(
                {
                    "difficulty": difficulty,
                    "condition": condition,
                    "metric": metric,
                    "n": len(group),
                    "mean": round(mean, 3),
                    "ci_low": round(lo, 3),
                    "ci_high": round(hi, 3),
                }
            )
    result = pd.DataFrame(rows)
    result["difficulty"] = pd.Categorical(
        result["difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    return result.sort_values(["difficulty", "condition", "metric"]).reset_index(
        drop=True
    )


def error_reduction_by_difficulty_reasoning(
    df: pd.DataFrame, metric: str = "errors", model_info: dict | None = None
) -> pd.DataFrame:
    """Mean per-run error reduction (before - after), broken down by prompt
    difficulty x condition (rows) and reasoning/non-reasoning model category
    (columns). Models missing from model_info.yaml are dropped, same as the
    plots that need reasoning-mode metadata.

    This is the central RQ1 validity check: if the validator's specific error
    list is doing real work (rather than just the act of reprompting), the
    feedback condition's reduction should exceed blind's within each
    difficulty x reasoning-category cell.
    """
    info = model_info if model_info is not None else load_model_info()
    df = df.copy()
    df["reasoning_category"] = df["model"].map(
        lambda m: reasoning_category(m, info)
    )
    df = df[df["reasoning_category"].notna()]

    initial = df[df["condition"] == "initial"][["run_id", metric]].rename(
        columns={metric: "before"}
    )

    rows = []
    for condition in ("feedback", "blind"):
        finals = _final_iteration_per_run(df, condition)[
            ["run_id", "difficulty", "reasoning_category", metric]
        ].rename(columns={metric: "after"})
        merged = finals.merge(initial, on="run_id")
        merged["reduction"] = merged["before"] - merged["after"]
        for (difficulty, category), group in merged.groupby(
            ["difficulty", "reasoning_category"]
        ):
            rows.append(
                {
                    "difficulty": difficulty,
                    "condition": condition,
                    "reasoning_category": category,
                    "n": len(group),
                    "mean_reduction": round(group["reduction"].mean(), 3),
                    "mean_before": round(group["before"].mean(), 3),
                    "mean_after": round(group["after"].mean(), 3),
                }
            )
    result = pd.DataFrame(rows)
    result["difficulty"] = pd.Categorical(
        result["difficulty"], categories=DIFFICULTY_ORDER, ordered=True
    )
    return result.sort_values(["difficulty", "condition"]).reset_index(drop=True)


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


def _category_counts(rows: pd.DataFrame) -> pd.Series:
    """Sum the {category: count} dicts in `rows["error_categories"]` into one
    Series. Note this bucket is computed upstream (util.validation.categorize_messages)
    over errors AND warnings together, not errors alone."""
    counts: dict[str, int] = {}
    for categories in rows["error_categories"]:
        for cat, n in categories.items():
            counts[cat] = counts.get(cat, 0) + n
    return pd.Series(counts, dtype="int64")


def top_error_categories(
    df: pd.DataFrame, condition: str = "initial", top_n: int = 10
) -> pd.Series:
    """Which error categories are most common in the given condition, across all runs."""
    return _category_counts(df[df["condition"] == condition]).sort_values(
        ascending=False
    ).head(top_n)


def error_category_before_after(
    df: pd.DataFrame, condition: str = "feedback", top_n: int = 12
) -> pd.DataFrame:
    """Before (initial generation) vs after (final `condition` iteration)
    counts per error category, sorted by how common the category was before
    reprompting. Answers which error types the guardrail is most/least
    effective at resolving (RQ1 paragraph 5)."""
    before = _category_counts(df[df["condition"] == "initial"])
    after = _category_counts(_final_iteration_per_run(df, condition))
    combined = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
    combined["delta"] = combined["before"] - combined["after"]
    combined["pct_resolved"] = combined.apply(
        lambda r: round(_pct_reduction(r["after"], r["before"]), 1)
        if r["before"]
        else float("nan"),
        axis=1,
    )
    combined = combined.sort_values("before", ascending=False).head(top_n)
    return combined.reset_index(names="category")


def _tokens_per_run(df: pd.DataFrame, condition: str) -> pd.Series:
    """Total tokens (prompt_eval_count + eval_count) per run_id, summed over
    the initial generation plus every iteration of `condition` — the full
    token bill for that condition's attempt at a run, not just its last
    iteration."""
    tokens = df["prompt_eval_count"].fillna(0) + df["eval_count"].fillna(0)
    tokens_df = df.assign(tokens=tokens)
    initial_tokens = tokens_df[tokens_df["condition"] == "initial"].set_index(
        "run_id"
    )["tokens"]
    cond_tokens = (
        tokens_df[tokens_df["condition"] == condition]
        .groupby("run_id")["tokens"]
        .sum()
    )
    return initial_tokens.add(cond_tokens, fill_value=0)


def token_cost_comparison(df: pd.DataFrame, metric: str = "errors") -> pd.DataFrame:
    """Per model (plus an 'ALL' pooled row): mean total tokens spent per run
    under feedback vs blind, and the cost-benefit ratio -- extra tokens spent
    per additional point of `metric` resolved by feedback over blind. Answers
    RQ2 paragraph 4: is feedback's extra token cost justified by its quality
    gain?"""
    feedback_tokens = _tokens_per_run(df, "feedback").rename("feedback_tokens")
    blind_tokens = _tokens_per_run(df, "blind").rename("blind_tokens")
    feedback_final = (
        _final_iteration_per_run(df, "feedback")
        .set_index("run_id")[metric]
        .rename("feedback_final")
    )
    blind_final = (
        _final_iteration_per_run(df, "blind")
        .set_index("run_id")[metric]
        .rename("blind_final")
    )
    model_of_run = df.drop_duplicates("run_id").set_index("run_id")["model"]

    merged = pd.concat(
        [model_of_run, feedback_tokens, blind_tokens, feedback_final, blind_final],
        axis=1,
    ).dropna()

    def _summarize(label: str, group: pd.DataFrame) -> dict:
        mean_feedback_tokens = group["feedback_tokens"].mean()
        mean_blind_tokens = group["blind_tokens"].mean()
        extra_tokens = mean_feedback_tokens - mean_blind_tokens
        extra_tokens_pct = (
            extra_tokens / mean_blind_tokens * 100 if mean_blind_tokens else 0.0
        )
        extra_resolved = group["blind_final"].mean() - group["feedback_final"].mean()
        ratio = extra_tokens / extra_resolved if extra_resolved > 0 else float("nan")
        return {
            "model": label,
            "n": len(group),
            "mean_feedback_tokens": round(mean_feedback_tokens, 1),
            "mean_blind_tokens": round(mean_blind_tokens, 1),
            "extra_tokens": round(extra_tokens, 1),
            "extra_tokens_pct": round(extra_tokens_pct, 1),
            "extra_resolved": round(extra_resolved, 3),
            "tokens_per_extra_resolved": round(ratio, 1)
            if not np.isnan(ratio)
            else None,
        }

    rows = [_summarize("ALL", merged)]
    for model, group in merged.groupby("model"):
        rows.append(_summarize(model, group))
    return pd.DataFrame(rows)


def main(results_csv: str) -> None:
    df = load_results(results_csv)

    print("\n=== Mean +/- 95% CI by model/condition (final iteration per run) ===")
    print(summarize_by_model_condition(df).to_string(index=False))

    print("\n=== Mean +/- 95% CI by difficulty/condition (final iteration per run) ===")
    print(summarize_by_difficulty_condition(df).to_string(index=False))

    for condition in ("feedback", "blind"):
        print(f"\n=== Paired before/after test — {condition} reprompting ===")
        for metric in METRICS:
            result = paired_before_after_test(df, condition, metric)
            if not result.empty:
                print(result.to_string(index=False))

    print("\n=== Aggregate (all models pooled) before/after test ===")
    for condition in ("feedback", "blind"):
        for metric in METRICS:
            print(f"-- {condition}/{metric}: {aggregate_before_after_test(df, condition, metric)}")

    print("\n=== Feedback vs blind ablation (paired, same initial HTML) ===")
    for metric in METRICS:
        result = feedback_vs_blind_test(df, metric)
        if not result.empty:
            print(result.to_string(index=False))

    print("\n=== Error reduction by difficulty x reasoning category ===")
    print(error_reduction_by_difficulty_reasoning(df).to_string(index=False))

    print("\n=== Top error categories in initial generations ===")
    print(top_error_categories(df, "initial").to_string())

    print("\n=== Error category breakdown: before vs after feedback reprompting ===")
    print(error_category_before_after(df).to_string(index=False))

    print("\n=== Token cost: feedback vs blind ===")
    print(token_cost_comparison(df).to_string(index=False))

    print("\n=== Iterations to convergence (feedback condition) ===")
    conv = iterations_to_convergence(df, "feedback")
    print(conv["iterations_to_convergence"].value_counts(dropna=False).sort_index().to_string())

    print("\n=== Iteration cutoff analysis — how many reprompts are worth it? ===")
    for condition in ("feedback", "blind"):
        result = iteration_cutoff_analysis(df, condition, "errors")
        if not result.empty:
            print(f"\n-- {condition} (by model) --")
            print(result.to_string(index=False))

    print("\n=== Iteration cutoff analysis — by difficulty tier ===")
    for condition in ("feedback", "blind"):
        result = iteration_cutoff_analysis(df, condition, "errors", group_by="difficulty")
        if not result.empty:
            print(f"\n-- {condition} (by difficulty) --")
            print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        default="results/experiments.csv",
        help="Path to the results CSV.",
    )
    args = parser.parse_args()
    main(args.results_csv)
