import re
import time

import ollama

SYSTEM_PROMPT = (
    "You generate HTML documents. Respond with the raw HTML code only: no "
    "markdown code fences, no explanation, no commentary before or after the "
    "code. Your entire response must be a single valid HTML document."
)

_FENCE_RE = re.compile(r"```(?:html|HTML)?\s*\n?(.*?)```", re.DOTALL)


def clean_html_output(raw: str) -> tuple[str, bool]:
    """Strip markdown code fences and non-HTML preamble/postamble from model output.

    Local instruction-following models frequently wrap HTML in ```html fences or
    add explanatory text despite being told not to. Validating that text directly
    would count formatting slip-ups as HTML standards violations, confounding the
    measurement of actual HTML quality. Returns (cleaned_text, was_modified).
    """
    text = raw.strip()

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate, candidate != text

    first_tag = text.find("<")
    last_tag = text.rfind(">")
    if first_tag == -1 or last_tag == -1 or last_tag < first_tag:
        return text, False

    candidate = text[first_tag : last_tag + 1]
    return candidate, candidate != text


def generate_html(
    model_name: str,
    prompt: str,
    output_path: str,
    temperature: float = 0.2,
    seed: int | None = None,
    print_code: bool = False,
) -> dict:
    """Stream a response from a local Ollama model, clean it, and save it as HTML.

    Returns a metadata dict (generation params, timing, token counts, whether the
    output needed cleaning) so callers can log it alongside validation results for
    reproducibility and cost/quality analysis.
    """
    raw_output = ""
    options = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed

    start = time.perf_counter()
    stream = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options=options,
        stream=True,
    )

    final_chunk = {}
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if print_code:
            print(content, end="", flush=True)
        raw_output += content
        if chunk.get("done"):
            final_chunk = chunk
    gen_time_s = time.perf_counter() - start

    cleaned_output, was_cleaned = clean_html_output(raw_output)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_output)
    print(f"\nHTML saved to {output_path}")

    return {
        "model_name": model_name,
        "temperature": temperature,
        "seed": seed,
        "was_cleaned": was_cleaned,
        "gen_time_s": gen_time_s,
        "prompt_eval_count": final_chunk.get("prompt_eval_count"),
        "eval_count": final_chunk.get("eval_count"),
        "total_duration_ns": final_chunk.get("total_duration"),
    }
