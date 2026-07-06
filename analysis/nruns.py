def check_prompt_completion(df, min_prompt=1, max_prompt=51):
	"""
	Check, for each unique model, whether prompts min_prompt to max_prompt
	are all present in the 'prompt_id' column (e.g. 'p1' to 'p51').

	Prints a line per model in the form:
	{MODEL_NAME}: fully completed ✅
	{MODEL_NAME}: ❌ (XX.X%)

	Parameters
	----------
	df : pd.DataFrame
		DataFrame containing at least 'model' and 'prompt_id' columns.
	min_prompt : int
		Lowest prompt number expected (default 1).
	max_prompt : int
		Highest prompt number expected (default 51).

	Returns
	-------
	pd.DataFrame
		Summary with columns: model, n_found, n_expected, pct_complete, missing_prompts
	"""
	expected_prompts = set(range(min_prompt, max_prompt + 1))

	rows = []
	for model, group in df.groupby("model"):
		found_prompts = set(group["prompt_id"].str.lstrip("p").astype(int).unique())

		missing = sorted(expected_prompts - found_prompts)
		n_found = len(expected_prompts & found_prompts)
		pct = 100 * n_found / len(expected_prompts)

		if not missing:
			pass
			print(f"✅ {model}")
		else:
			print(
				# f"{model}\t{pct:.1f}% ({n_found}/51)"
				f"❌ {model}\t\t{pct:.1f}% ({n_found}/51"
			)

		rows.append(
			{
				"model": model,
				"n_found": n_found,
				"n_expected": len(expected_prompts),
				"pct_complete": pct,
				"missing_prompts": missing,
			}
		)

	return pd.DataFrame(rows)


if __name__ == "__main__":
	import pandas as pd

	df = pd.read_csv("results/experiments.csv")

	summary = check_prompt_completion(df)
	summary
