import argparse
import os
import time

from util.config import LOCAL_VALIDATOR_URL
from util.generation import generate_html
from util.pipeline import print_comparison, run_reprompt_loop, validate_and_parse
from util.print_functions import section_print
from util.prompts import choose_prompt
from util.validation import summarise_validation

parser = argparse.ArgumentParser(
	description="Generate HTML, validate it, reprompt to fix issues, then validate again."
)
parser.add_argument(
	"--validate-only",
	action="store_true",
	help="Skip generation and only validate an existing HTML file.",
)
parser.add_argument(
	"--validate-and-regenerate",
	nargs="?",
	type=int,
	const=1,
	default=None,
	metavar="N",
	help="Validate an existing HTML file, then automatically build a reprompt and re-generate N times (default: 1).",
)

args = parser.parse_args()

DIRS = {
	"html": "./html",
	"html_reprompt": "./html/reprompt",
	"validation": "./validation",
	"validation_reprompt": "./validation/reprompt",
}

MODEL_NAME = "gemma4:e2b"  # qwen3:8b, gemma4:e2b, qwen3.5:9b (❌ qwen2.5-coder:14b)
PROMPTS_PATH = "./prompts/prompts.json"
VALIDATE_ONLY_FILE = "./html/reprompt/generated_2026-03-17_16-25-29.html"

if __name__ == "__main__":
	validator = LOCAL_VALIDATOR_URL
	timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

	if args.validate_only:
		# ── Validate a single existing file, no generation ────────────────────
		os.makedirs(DIRS["validation"], exist_ok=True)
		validate_and_parse(
			html_path=VALIDATE_ONLY_FILE,
			validator=validator,
			output_path=f"{DIRS['validation']}/validation_{timestamp}.json",
		)

	elif args.validate_and_regenerate is not None:
		# ── Validate existing file, reprompt, re-generate N times ─────────────
		for d in DIRS.values():
			os.makedirs(d, exist_ok=True)

		section_print("STEP 1 — Validating existing HTML")
		initial_validation_path = f"{DIRS['validation']}/validation_{timestamp}.json"
		validate_and_parse(VALIDATE_ONLY_FILE, validator, initial_validation_path)
		before = summarise_validation(initial_validation_path)

		n = args.validate_and_regenerate
		_, final_validation_path = run_reprompt_loop(
			html_path=VALIDATE_ONLY_FILE,
			validation_path=initial_validation_path,
			prompt=choose_prompt(PROMPTS_PATH),
			n_iterations=n,
			model_name=MODEL_NAME,
			html_reprompt_dir=DIRS["html_reprompt"],
			validation_reprompt_dir=DIRS["validation_reprompt"],
			validator=validator,
			timestamp=timestamp,
		)
		print_comparison(
			before, after=summarise_validation(final_validation_path), n_iterations=n
		)

	else:
		# ── Generate → validate → reprompt → validate ─────────────────────────
		for d in DIRS.values():
			os.makedirs(d, exist_ok=True)

		section_print("STEP 1 — Generating initial HTML")
		initial_html_path = f"{DIRS['html']}/generated_{timestamp}.html"
		prompt = choose_prompt(PROMPTS_PATH)
		generate_html(
			model_name=MODEL_NAME,
			prompt=prompt["text"],
			output_path=initial_html_path,
		)

		section_print("STEP 2 — Validating initial HTML")
		initial_validation_path = f"{DIRS['validation']}/validation_{timestamp}.json"
		validate_and_parse(initial_html_path, validator, initial_validation_path)
		before = summarise_validation(initial_validation_path)

		_, final_validation_path = run_reprompt_loop(
			html_path=initial_html_path,
			validation_path=initial_validation_path,
			prompt=prompt["text"],
			n_iterations=1,
			model_name=MODEL_NAME,
			html_reprompt_dir=DIRS["html_reprompt"],
			validation_reprompt_dir=DIRS["validation_reprompt"],
			validator=validator,
			timestamp=timestamp,
		)
		print_comparison(before, after=summarise_validation(final_validation_path))
