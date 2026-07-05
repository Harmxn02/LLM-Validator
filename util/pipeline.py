from util.generation import generate_html
from util.print_functions import section_print
from util.prompts import build_reprompt
from util.validation import (
    parse_validation_results,
    validate_html,
    summarise_validation,
)


def validate_and_parse(html_path, validator, output_path):
    """Validate an HTML file, print a summary, and return the parsed summary dict."""
    validate_html(
        html_path=html_path,
        validator=validator,
        validation_output_path=output_path,
    )
    parse_validation_results(output_path)
    return summarise_validation(output_path)


def print_comparison(before, after, n_iterations=1):
    """Print a before/after comparison table of validation error/warning/info counts."""
    label = (
        f"REPROMPT COMPARISON (after {n_iterations} iteration(s))"
        if n_iterations > 1
        else "REPROMPT COMPARISON"
    )
    section_print(label)
    print(f"{'Metric':<12} {'Before':>8} {'After':>8} {'Δ':>8}")
    print(f"{'-' * 40}")
    for key in ("errors", "warnings", "infos"):
        delta = after[key] - before[key]
        sign = "+" if delta > 0 else ""
        print(
            f"{key.capitalize():<12} {before[key]:>8} {after[key]:>8} {sign + str(delta):>8}"
        )
    print(f"{'=' * 50}\n")


def run_reprompt_loop(
    html_path,
    validation_path,
    prompt,
    n_iterations,
    model_name,
    html_reprompt_dir,
    validation_reprompt_dir,
    validator,
    timestamp,
    temperature=0.2,
    seed=None,
    blind=False,
    on_iteration=None,
):
    """
    Run the reprompt → generate → validate cycle up to N times, stopping early
    if all errors, warnings, and infos are resolved.

    `blind` withholds the validator's error list from the reprompt (ablation
    control condition). `on_iteration(iteration, gen_metadata, summary)`, if
    given, is called after each iteration so callers (e.g. the batch runner)
    can log full per-iteration metadata without this function needing to know
    about any particular results store.

    Returns a tuple of (final_html_path, final_validation_path).
    """
    current_html_path = html_path
    current_validation_path = validation_path

    for i in range(1, n_iterations + 1):
        section_print(
            f"ITERATION {i}/{n_iterations} — Re-generating HTML using validation feedback"
        )
        reprompt = build_reprompt(
            original_html_path=current_html_path,
            validation_path=current_validation_path,
            original_prompt=prompt,
            blind=blind,
        )
        current_html_path = f"{html_reprompt_dir}/reprompted_{timestamp}_iter{i}.html"
        gen_metadata = generate_html(
            model_name=model_name,
            prompt=reprompt,
            output_path=current_html_path,
            temperature=temperature,
            seed=seed,
        )

        section_print(f"ITERATION {i}/{n_iterations} — Validating reprompted HTML")
        current_validation_path = (
            f"{validation_reprompt_dir}/validation_reprompted_{timestamp}_iter{i}.json"
        )
        summary = validate_and_parse(
            current_html_path, validator, current_validation_path
        )

        if on_iteration is not None:
            on_iteration(i, gen_metadata, summary)

        if all(summary[key] == 0 for key in ("errors", "warnings", "infos")):
            section_print(
                f"ITERATION {i}/{n_iterations} — All issues resolved, stopping early"
            )
            break

    return current_html_path, current_validation_path
