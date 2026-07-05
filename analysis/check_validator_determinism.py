"""Check whether the W3C validator returns consistent results for the same input.

Before trusting any before/after error-count delta as evidence the *model*
improved the HTML, we need to rule out the validator itself being a source of
noise (e.g. Nu Checker's HTML5 parser has some heuristics that could plausibly
be order/timing sensitive). This submits the same file N times and reports
whether the error/warning/info counts — and the exact message set — are
identical every time.

Usage:
    uv run python -m analysis.check_validator_determinism --html-file path/to.html --n 10
"""

import argparse
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.config import LOCAL_VALIDATOR_URL
from util.validation import summarise_validation, validate_html


def check_determinism(html_path: str, n: int, validator: str) -> None:
    summaries = []
    message_sets = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(n):
            out_path = os.path.join(tmp_dir, f"run_{i}.json")
            validate_html(html_path, validator, out_path)
            summary = summarise_validation(out_path)
            summaries.append(summary)

            with open(out_path) as f:
                messages = json.load(f).get("messages", [])
            message_sets.append(frozenset(m["message"] for m in messages))

    print(f"\n{'=' * 60}")
    print(f"Validator determinism check — {n} runs on {html_path}")
    print(f"{'=' * 60}")

    for key in ("errors", "warnings", "infos"):
        values = [s[key] for s in summaries]
        spread = max(values) - min(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        flag = "STABLE" if spread == 0 else "VARIES"
        print(
            f"{key:<10} min={min(values):<3} max={max(values):<3} "
            f"mean={statistics.mean(values):<6.2f} stdev={stdev:<6.3f} [{flag}]"
        )

    unique_message_sets = set(message_sets)
    if len(unique_message_sets) == 1:
        print("\nExact message set was IDENTICAL across all runs.")
    else:
        print(
            f"\nWARNING: message set differed across runs "
            f"({len(unique_message_sets)} distinct sets out of {n} runs)."
        )
        baseline = message_sets[0]
        for i, ms in enumerate(message_sets[1:], start=1):
            added = ms - baseline
            removed = baseline - ms
            if added or removed:
                print(f"  run 0 vs run {i}: +{len(added)} new, -{len(removed)} missing")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-file", required=True, help="HTML file to repeatedly validate."
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of repeat validations (default: 10)."
    )
    parser.add_argument(
        "--validator", default=LOCAL_VALIDATOR_URL, help="Validator endpoint URL."
    )
    args = parser.parse_args()

    check_determinism(args.html_file, args.n, args.validator)
