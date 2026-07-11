import os

import yaml

MODEL_INFO_PATH = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	"config",
	"model_info.yaml",
)

THINKING_LABELS = {True: "Reasoning", False: "Non-reasoning"}


def load_model_info(path: str = MODEL_INFO_PATH) -> dict:
	"""Static per-model metadata that isn't derivable from the results CSV:
	billion_params (total, on-disk parameter count) and native_thinking_mode.

	native_thinking_mode means the model ships with a documented, toggleable
	extended-thinking / chain-of-thought mode (e.g. Qwen3's hybrid thinking
	mode) — a specific, checkable architectural feature, unlike the vaguer
	"does this model reason" framing.
	"""
	with open(path) as f:
		return yaml.safe_load(f)


def reasoning_category(model: str, model_info: dict | None = None) -> str | None:
	"""'Reasoning' / 'Non-reasoning' label for a model, or None if the model
	is missing from model_info.yaml (e.g. deliberately excluded, like
	smollm2:135m)."""
	info = model_info if model_info is not None else load_model_info()
	entry = info.get(model)
	if entry is None:
		return None
	return THINKING_LABELS[entry["native_thinking_mode"]]
