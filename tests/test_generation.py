from util.generation import clean_html_output


def test_clean_html_output_strips_fences():
	raw = "```html\n<!DOCTYPE html><html><body>hi</body></html>\n```"
	cleaned, was_cleaned = clean_html_output(raw)
	assert cleaned == "<!DOCTYPE html><html><body>hi</body></html>"
	assert was_cleaned is True


def test_clean_html_output_strips_preamble_without_fences():
	raw = "Sure! Here's the HTML:\n<!DOCTYPE html><html></html>\nHope that helps!"
	cleaned, was_cleaned = clean_html_output(raw)
	assert cleaned == "<!DOCTYPE html><html></html>"
	assert was_cleaned is True


def test_clean_html_output_noop_on_clean_input():
	raw = "<!DOCTYPE html><html><body>hi</body></html>"
	cleaned, was_cleaned = clean_html_output(raw)
	assert cleaned == raw
	assert was_cleaned is False


def test_clean_html_output_handles_empty_string():
	cleaned, was_cleaned = clean_html_output("")
	assert cleaned == ""
	assert was_cleaned is False
