import csv
import json
import os

import pandas as pd

FIELDNAMES = [
	"run_id",
	"timestamp",
	"model",
	"prompt_id",
	"difficulty",
	"condition",
	"trial",
	"iteration",
	"temperature",
	"seed",
	"errors",
	"warnings",
	"infos",
	"error_categories",
	"gen_time_s",
	"prompt_eval_count",
	"eval_count",
	"was_cleaned",
	"html_path",
	"validation_path",
]


def make_run_record(
	run_id: str,
	timestamp: str,
	model: str,
	prompt_id: str,
	difficulty: str,
	condition: str,
	trial: int,
	iteration: int,
	temperature: float,
	seed: int | None,
	errors: int,
	warnings: int,
	infos: int,
	error_categories: dict,
	gen_time_s: float | None,
	prompt_eval_count: int | None,
	eval_count: int | None,
	was_cleaned: bool,
	html_path: str,
	validation_path: str,
) -> dict:
	"""Build one structured run record with an explicit, fixed schema.

	Using named parameters (rather than an arbitrary dict) makes it obvious at
	every call site which fields a run must report, so a run can never be
	silently logged with missing columns.
	"""
	return {
		"run_id": run_id,
		"timestamp": timestamp,
		"model": model,
		"prompt_id": prompt_id,
		"difficulty": difficulty,
		"condition": condition,
		"trial": trial,
		"iteration": iteration,
		"temperature": temperature,
		"seed": seed,
		"errors": errors,
		"warnings": warnings,
		"infos": infos,
		"error_categories": json.dumps(error_categories),
		"gen_time_s": gen_time_s,
		"prompt_eval_count": prompt_eval_count,
		"eval_count": eval_count,
		"was_cleaned": was_cleaned,
		"html_path": html_path,
		"validation_path": validation_path,
	}


def append_run_record(csv_path: str, record: dict) -> None:
	"""Append a single run record as a row, writing the header on first write."""
	parent = os.path.dirname(csv_path)
	if parent:
		os.makedirs(parent, exist_ok=True)

	file_exists = os.path.isfile(csv_path)
	with open(csv_path, "a", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
		if not file_exists:
			writer.writeheader()
		writer.writerow(record)


def load_results(csv_path: str) -> pd.DataFrame:
	"""Load the results CSV into a DataFrame, parsing error_categories back to dict."""
	df = pd.read_csv(csv_path)
	if "error_categories" in df.columns:
		df["error_categories"] = df["error_categories"].apply(
			lambda s: json.loads(s) if isinstance(s, str) else {}
		)
	return df
