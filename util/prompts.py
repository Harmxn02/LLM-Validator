import json
import random


def load_prompts(file_path: str = "prompts.json") -> list[dict]:
	"""Load the full structured prompt list: [{id, difficulty, text}, ...]."""
	with open(file_path, "r") as f:
		all_data = json.load(f)
	return all_data["prompts"]


def choose_prompt(file_path: str = "prompts.json", rng: random.Random | None = None) -> dict:
	"""Return a single randomly selected prompt dict ({id, difficulty, text}).

	Pass an explicit `rng` (e.g. random.Random(seed)) for reproducible selection;
	otherwise falls back to the global random module.
	"""
	prompts = load_prompts(file_path)
	chooser = rng.choice if rng is not None else random.choice
	return chooser(prompts)


def iter_prompts(file_path: str = "prompts.json", difficulty: str | None = None) -> list[dict]:
	"""Return the fixed, ordered list of prompts (optionally filtered by difficulty).

	Used by the batch runner so every model/condition is evaluated on the exact
	same prompt set, instead of drawing an independent random prompt per run.
	"""
	prompts = load_prompts(file_path)
	if difficulty is not None:
		prompts = [p for p in prompts if p["difficulty"] == difficulty]
	return prompts


def build_reprompt(
	original_html_path: str,
	validation_path: str,
	original_prompt: str,
	blind: bool = False,
) -> str:
	"""Construct a follow-up prompt asking the model to fix the generated HTML.

	Reads the previously generated HTML and the saved validation JSON, collects all
	errors and warnings, and combines them with the original prompt into a single
	instruction string that can be passed directly to generate_html.

	If `blind` is True, the validator's error list is withheld and the model is only
	told to review and fix any issues itself. This is the ablation control condition
	used to measure how much of the improvement is actually attributable to the
	validator feedback, as opposed to just a second generation attempt.
	"""
	with open(original_html_path, "r", encoding="utf-8") as f:
		html_content = f.read()

	if blind:
		return (
			f"The following HTML was originally generated for this request:\n"
			f"{original_prompt}\n\n"
			f"Here is the generated HTML:\n"
			f"{html_content}\n\n"
			f"Please review this HTML for any mistakes or standards violations and fix "
			f"them. Return only the corrected, complete HTML document. Do not include "
			f"any explanation or markdown fences."
		)

	with open(validation_path, "r", encoding="utf-8") as f:
		validation_result = json.load(f)

	messages = validation_result.get("messages", [])
	errors = [m for m in messages if m["type"] == "error"]
	warnings = [
		m for m in messages if m["type"] == "info" and m.get("subType") == "warning"
	]

	issue_lines = []
	for m in errors + warnings:
		line = m.get("lastLine", "?")
		col = m.get("lastColumn", "?")
		label = "ERROR" if m["type"] == "error" else "WARNING"
		issue_lines.append(f"[{label}] Line {line}, Col {col}: {m['message']}")

	issues_text = (
		"\n".join(issue_lines) if issue_lines else "No errors or warnings were found."
	)

	return (
		f"The following HTML was originally generated for this request:\n"
		f"{original_prompt}\n\n"
		f"Here is the generated HTML:\n"
		f"{html_content}\n\n"
		f"The W3C HTML validator reported these issues:\n"
		f"{issues_text}\n\n"
		f"Please fix every issue listed above and return only the corrected, "
		f"complete HTML document. Do not include any explanation or markdown fences."
	)
