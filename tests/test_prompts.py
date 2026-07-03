import json
import random

from util.prompts import build_reprompt, choose_prompt, iter_prompts


def _write_prompts_file(tmp_path):
	data = {
		"prompts": [
			{"id": "p1", "difficulty": "simple", "text": "A"},
			{"id": "p2", "difficulty": "medium", "text": "B"},
			{"id": "p3", "difficulty": "difficult", "text": "C"},
		]
	}
	path = tmp_path / "prompts.json"
	path.write_text(json.dumps(data))
	return str(path)


def test_iter_prompts_returns_all_in_order(tmp_path):
	path = _write_prompts_file(tmp_path)
	assert [p["id"] for p in iter_prompts(path)] == ["p1", "p2", "p3"]


def test_iter_prompts_filters_by_difficulty(tmp_path):
	path = _write_prompts_file(tmp_path)
	assert [p["id"] for p in iter_prompts(path, difficulty="medium")] == ["p2"]


def test_choose_prompt_is_deterministic_with_seeded_rng(tmp_path):
	path = _write_prompts_file(tmp_path)
	chosen_1 = choose_prompt(path, rng=random.Random(42))
	chosen_2 = choose_prompt(path, rng=random.Random(42))
	assert chosen_1 == chosen_2


def test_build_reprompt_feedback_includes_validator_errors(tmp_path):
	html_path = tmp_path / "gen.html"
	html_path.write_text("<html></html>")
	validation_path = tmp_path / "validation.json"
	validation_path.write_text(
		json.dumps({"messages": [{"type": "error", "message": "Stray tag", "lastLine": 1, "lastColumn": 2}]})
	)

	reprompt = build_reprompt(str(html_path), str(validation_path), "Build a page", blind=False)

	assert "Stray tag" in reprompt
	assert "W3C HTML validator reported" in reprompt


def test_build_reprompt_blind_excludes_validator_errors(tmp_path):
	html_path = tmp_path / "gen.html"
	html_path.write_text("<html></html>")

	reprompt = build_reprompt(str(html_path), "unused.json", "Build a page", blind=True)

	assert "W3C HTML validator" not in reprompt
	assert "review this HTML" in reprompt
