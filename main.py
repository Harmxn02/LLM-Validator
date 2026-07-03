import argparse
import os
import time

from util.config import LOCAL_VALIDATOR_URL
from util.generation import generate_html
from util.pipeline import print_comparison, run_reprompt_loop, validate_and_parse
from util.print_functions import section_print
from util.prompts import choose_prompt

parser = argparse.ArgumentParser(
	description="Generate HTML, validate it, reprompt to fix issues, then validate again."
)
parser.add_argument(
	"--model",
	default="qwen3:8b",
	help="Ollama model name to use (default: qwen3:8b).",
)
parser.add_argument(
	"--prompts-path",
	default="./prompts/prompts.json",
	help="Path to the prompts JSON file (default: ./prompts/prompts.json).",
)
parser.add_argument(
	"--temperature",
	type=float,
	default=0.2,
	help="Sampling temperature passed to Ollama (default: 0.2).",
)
parser.add_argument(
	"--seed",
	type=int,
	default=None,
	help="Sampling seed passed to Ollama, for reproducible single runs (default: unset).",
)
parser.add_argument(
	"--blind",
	action="store_true",
	help="Reprompt without giving the model the validator's error list (ablation control condition).",
)
parser.add_argument(
	"--html-file",
	default=None,
	help="Existing HTML file to validate. Required with --validate-only or --validate-and-regenerate.",
)
parser.add_argument(
	"--validate-only",
	action="store_true",
	help="Skip generation and only validate an existing HTML file (requires --html-file).",
)
parser.add_argument(
	"--validate-and-regenerate",
	nargs="?",
	type=int,
	const=1,
	default=None,
	metavar="N",
	help="Validate an existing HTML file (requires --html-file), then automatically build a reprompt and re-generate N times (default: 1).",
)

args = parser.parse_args()

DIRS = {
	"html": "./html",
	"html_reprompt": "./html/reprompt",
	"validation": "./validation",
	"validation_reprompt": "./validation/reprompt",
}

if __name__ == "__main__":
	validator = LOCAL_VALIDATOR_URL
	timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

	if args.validate_only:
		# ── Validate a single existing file, no generation ────────────────────
		if not args.html_file:
			parser.error("--validate-only requires --html-file")
		os.makedirs(DIRS["validation"], exist_ok=True)
		validate_and_parse(
			html_path=args.html_file,
			validator=validator,
			output_path=f"{DIRS['validation']}/validation_{timestamp}.json",
		)

	elif args.validate_and_regenerate is not None:
		# ── Validate existing file, reprompt, re-generate N times ─────────────
		if not args.html_file:
			parser.error("--validate-and-regenerate requires --html-file")
		for d in DIRS.values():
			os.makedirs(d, exist_ok=True)

		section_print("STEP 1 — Validating existing HTML")
		initial_validation_path = f"{DIRS['validation']}/validation_{timestamp}.json"
		before = validate_and_parse(args.html_file, validator, initial_validation_path)

		n = args.validate_and_regenerate
		final_summary = {}
		_, final_validation_path = run_reprompt_loop(
			html_path=args.html_file,
			validation_path=initial_validation_path,
			prompt=choose_prompt(args.prompts_path)["text"],
			n_iterations=n,
			model_name=args.model,
			html_reprompt_dir=DIRS["html_reprompt"],
			validation_reprompt_dir=DIRS["validation_reprompt"],
			validator=validator,
			timestamp=timestamp,
			temperature=args.temperature,
			seed=args.seed,
			blind=args.blind,
			on_iteration=lambda i, gen_metadata, summary: final_summary.update(summary),
		)
		print_comparison(before, after=final_summary, n_iterations=n)

	else:
		# ── Generate → validate → reprompt → validate ─────────────────────────
		for d in DIRS.values():
			os.makedirs(d, exist_ok=True)

		section_print("STEP 1 — Generating initial HTML")
		initial_html_path = f"{DIRS['html']}/generated_{timestamp}.html"
		prompt = choose_prompt(args.prompts_path)
		generate_html(
			model_name=args.model,
			prompt=prompt["text"],
			output_path=initial_html_path,
			temperature=args.temperature,
			seed=args.seed,
		)

		section_print("STEP 2 — Validating initial HTML")
		initial_validation_path = f"{DIRS['validation']}/validation_{timestamp}.json"
		before = validate_and_parse(initial_html_path, validator, initial_validation_path)

		final_summary = {}
		_, final_validation_path = run_reprompt_loop(
			html_path=initial_html_path,
			validation_path=initial_validation_path,
			prompt=prompt["text"],
			n_iterations=1,
			model_name=args.model,
			html_reprompt_dir=DIRS["html_reprompt"],
			validation_reprompt_dir=DIRS["validation_reprompt"],
			validator=validator,
			timestamp=timestamp,
			temperature=args.temperature,
			seed=args.seed,
			blind=args.blind,
			on_iteration=lambda i, gen_metadata, summary: final_summary.update(summary),
		)
		print_comparison(before, after=final_summary)
