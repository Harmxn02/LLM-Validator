"""Batch experiment runner.

Sweeps every (model x prompt x trial) combination through a single shared
initial generation, then runs both the feedback and blind-ablation reprompt
loops from that *same* initial artifact so the two conditions are directly
comparable. Every iteration of every run is logged as one row in the
structured results CSV (see util/results_store.py) for later statistical
analysis, instead of scattered one-off JSON files.

Usage:
    uv run python -m experiments.run_batch [--config config/experiment.json] [--dry-run]
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.config import LOCAL_VALIDATOR_URL
from util.generation import generate_html
from util.pipeline import run_reprompt_loop, validate_and_parse
from util.print_functions import section_print
from util.prompts import iter_prompts
from util.results_store import append_run_record, make_run_record


def derive_seed(base_seed: int, *parts: str) -> int:
    """Deterministically derive a per-run seed from base_seed + run identity.

    Using a stable hash (not Python's randomized built-in hash()) means the
    exact same batch config always reproduces the exact same seed per run,
    while still varying seeds across trials/prompts/models/conditions so
    repeated trials sample genuinely different completions.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return base_seed + (int(digest[:8], 16) % 100_000)


def safe_name(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_batch(config: dict, dry_run: bool = False) -> None:
    models = config["models"]
    difficulties = config["difficulties"]
    conditions = config["conditions"]
    trials = config["trials"]
    iterations = config["iterations"]
    temperature = config["temperature"]
    base_seed = config["base_seed"]
    dirs = config["output_dirs"]
    results_csv = config["results_csv"]

    prompts = [
        p
        for d in difficulties
        for p in iter_prompts(config["prompts_path"], difficulty=d)
    ]

    total_runs = len(models) * len(prompts) * trials
    planned = 0

    if not dry_run:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

    for model in models:
        for prompt in prompts:
            for trial in range(1, trials + 1):
                planned += 1
                run_key = f"{safe_name(model)}_{prompt['id']}_t{trial}"
                seed = derive_seed(base_seed, model, prompt["id"], trial)

                section_print(
                    f"[{planned}/{total_runs}] {model} | {prompt['id']} ({prompt['difficulty']}) | trial {trial} | seed={seed}"
                )
                if dry_run:
                    continue

                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                initial_html_path = f"{dirs['html']}/{run_key}_{timestamp}.html"
                initial_validation_path = (
                    f"{dirs['validation']}/{run_key}_{timestamp}.json"
                )

                gen_metadata0 = generate_html(
                    model_name=model,
                    prompt=prompt["text"],
                    output_path=initial_html_path,
                    temperature=temperature,
                    seed=seed,
                )
                summary0 = validate_and_parse(
                    initial_html_path, LOCAL_VALIDATOR_URL, initial_validation_path
                )

                append_run_record(
                    results_csv,
                    make_run_record(
                        run_id=run_key,
                        timestamp=timestamp,
                        model=model,
                        prompt_id=prompt["id"],
                        difficulty=prompt["difficulty"],
                        condition="initial",
                        trial=trial,
                        iteration=0,
                        temperature=temperature,
                        seed=seed,
                        errors=summary0["errors"],
                        warnings=summary0["warnings"],
                        infos=summary0["infos"],
                        error_categories=summary0["categories"],
                        gen_time_s=gen_metadata0["gen_time_s"],
                        prompt_eval_count=gen_metadata0["prompt_eval_count"],
                        eval_count=gen_metadata0["eval_count"],
                        was_cleaned=gen_metadata0["was_cleaned"],
                        html_path=initial_html_path,
                        validation_path=initial_validation_path,
                    ),
                )

                already_perfect = all(
                    summary0[k] == 0 for k in ("errors", "warnings", "infos")
                )
                if already_perfect:
                    section_print(
                        f"{run_key} — initial generation already has 0 issues, skipping reprompt loop for both conditions"
                    )
                    continue

                for condition in conditions:

                    def on_iteration(i, gen_metadata, summary, condition=condition):
                        append_run_record(
                            results_csv,
                            make_run_record(
                                run_id=run_key,
                                timestamp=timestamp,
                                model=model,
                                prompt_id=prompt["id"],
                                difficulty=prompt["difficulty"],
                                condition=condition,
                                trial=trial,
                                iteration=i,
                                temperature=temperature,
                                seed=seed,
                                errors=summary["errors"],
                                warnings=summary["warnings"],
                                infos=summary["infos"],
                                error_categories=summary["categories"],
                                gen_time_s=gen_metadata["gen_time_s"],
                                prompt_eval_count=gen_metadata["prompt_eval_count"],
                                eval_count=gen_metadata["eval_count"],
                                was_cleaned=gen_metadata["was_cleaned"],
                                html_path=f"{dirs['html_reprompt']}/{run_key}_{condition}_{timestamp}_iter{i}.html",
                                validation_path=f"{dirs['validation_reprompt']}/{run_key}_{condition}_{timestamp}_iter{i}.json",
                            ),
                        )

                    run_reprompt_loop(
                        html_path=initial_html_path,
                        validation_path=initial_validation_path,
                        prompt=prompt["text"],
                        n_iterations=iterations,
                        model_name=model,
                        html_reprompt_dir=dirs["html_reprompt"],
                        validation_reprompt_dir=dirs["validation_reprompt"],
                        validator=LOCAL_VALIDATOR_URL,
                        timestamp=f"{run_key}_{condition}_{timestamp}",
                        temperature=temperature,
                        seed=seed,
                        blind=(condition == "blind"),
                        on_iteration=on_iteration,
                    )

    section_print(
        f"Batch complete — {planned} (model, prompt, trial) runs. Results in {results_csv}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full model x prompt x trial x condition experiment sweep."
    )
    parser.add_argument(
        "--config",
        default="config/experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help='Comma-separated model names to override the config\'s "models" list, e.g. '
        "--models qwen3:8b or --models qwen3:8b,gemma3:4b. Useful for adding models to an "
        "existing results CSV one (or a few) at a time without re-running models already in it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run matrix without generating or validating anything.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.models:
        config["models"] = [m.strip() for m in args.models.split(",") if m.strip()]

    run_batch(config, dry_run=args.dry_run)
