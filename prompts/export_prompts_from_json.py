import json


def json_to_latex_table(data, label="tab:prompts", caption="Full prompt dataset"):
	"""Converts the prompt dataset JSON into a LaTeX longtable string.

	Escapes LaTeX special characters in the difficulty and prompt text
	fields, then builds a formatted longtable with ID, Difficulty, and
	Prompt columns, ready to paste straight into the appendix.

	Parameters
	----------
	data : dict
		Parsed JSON containing a "prompts" list of dicts with keys
		"id", "difficulty", and "text".
	label : str
		LaTeX label for cross-referencing the table.
	caption : str
		Caption text for the table.

	Returns
	-------
	str
		A complete LaTeX longtable environment as a string.
	"""
	specials = {
		"&": r"\&",
		"%": r"\%",
		"$": r"\$",
		"#": r"\#",
		"_": r"\_",
		"{": r"\{",
		"}": r"\}",
		"~": r"\textasciitilde{}",
		"^": r"\textasciicircum{}",
		"\\": r"\textbackslash{}",
		"<": r"$<$",
		">": r"$>$",
	}

	def escape(text):
		return "".join(specials.get(ch, ch) for ch in text)

	rows = []
	for prompt in data["prompts"]:
		pid = escape(prompt["id"])
		difficulty = escape(prompt["difficulty"].capitalize())
		text = escape(prompt["text"])
		rows.append(f"{pid} & {difficulty} & {text} \\\\")

	body = "\n".join(rows)

	table = rf"""\begin{{longtable}}{{@{{}} l l p{{10cm}} @{{}}}}
\caption{{{caption}}} \label{{{label}}} \\
\toprule
\textbf{{ID}} & \textbf{{Difficulty}} & \textbf{{Prompt}} \\
\midrule
\endfirsthead

\multicolumn{{3}}{{c}}{{\tablename\ \thetable{{}} -- continued}} \\
\toprule
\textbf{{ID}} & \textbf{{Difficulty}} & \textbf{{Prompt}} \\
\midrule
\endhead

\midrule
\multicolumn{{3}}{{r}}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

{body}
\end{{longtable}}"""
	return table


if __name__ == "__main__":
	data = json.loads(open("prompts.json").read())
	print(json_to_latex_table(data, label="tab:prompts", caption="Full prompt dataset"))
