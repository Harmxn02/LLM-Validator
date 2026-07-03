<!-- markdownlint-disable MD010 -->

# LLM Generated HTML Code Validator Loop

- [1 iteratie, werkt volledig](#voorbeeld-van-feedback-loop-die-compleet-werkt)
- [1 iteratie, werkt gedeeltelijk](#voorbeeld-van-feedback-loop-die-gedeeltelijk-werkt)
- [**3 iteraties, werkt volledig**](#voorbeeld-van-feedback-loop-met-meerdere-iteraties-die-volledig-werkt)

## Rigorous batch experiments

The sections below document early manual, single-run exploration. For anything
used as evidence in the research write-up, use the batch pipeline instead —
it fixes several confounds in the manual flow (random per-run prompts,
markdown fences leaking into validation, mixed cloud/local validators, no
repeated trials) and adds a blind-reprompt ablation as a control condition.

```bash
uv sync                                                   # install deps
docker run -p 8888:8888 ghcr.io/validator/validator:latest --port 8888
ollama serve                                              # in another terminal

# edit config/experiment.json to pick models/trials/iterations, then:
uv run python -m experiments.run_batch                    # sweep model x prompt x trial x condition
uv run python -m analysis.stats                            # significance tests + summary tables
uv run python -m analysis.plots                             # convergence/comparison/tradeoff plots
uv run python -m analysis.check_validator_determinism --html-file some.html  # sanity-check the validator itself
uv run pytest                                               # unit tests
```

Every run (each reprompt iteration, for every model/prompt/trial/condition)
is appended as one row to `results/experiments.csv` for analysis — see
`util/results_store.py` for the schema.

## Voorbeeld van feedback-loop die compleet werkt

Bij deze run werden initieel 5 errors gevonden, maar na het toepassen van 1 feedback-loop, zijn alle errors opgelost en zijn er 0 warnings. De feedback-loop lijkt hier dus volledig te werken.

Je kan de volledige output hieronder bekijken, of in dit bestand: [./logs/Full.txt](./logs/Full.txt)

```powershell
PS C:\Users\Harman\<project_path>\LLM Validator> python main.py --local

==================================================
STEP 1 — Generating initial HTML
==================================================


HTML saved to ./html/generated_2026-03-17_16-25-29.html

==================================================
STEP 2 — Validating initial HTML
==================================================

Validation results saved to ./validation/validation_2026-03-17_16-25-29.json
==================================================
Errors:   5
Warnings: 0
Info:     0
==================================================

[ERROR] Line 1, Col 142: Non-space characters found without seeing a doctype first. Expected “<!DOCTYPE html>”.
[ERROR] Line 1, Col 142: Element “head” is missing a required instance of child element “title”.
[ERROR] Line 4, Col 15: Stray doctype.
[ERROR] Line 5, Col 16: Stray start tag “html”.
[ERROR] Line 5, Col 16: Cannot recover after last error. Any further errors will be ignored.

==================================================
STEP 3 — Re-generating HTML using validation feedback
==================================================


HTML saved to ./html/reprompt/reprompted_2026-03-17_16-25-29.html

==================================================
STEP 4 — Validating reprompted HTML
==================================================

Validation results saved to ./validation/reprompt/validation_reprompted_2026-03-17_16-25-29.json
==================================================
Errors:   0
Warnings: 0
Info:     0
==================================================


==================================================
REPROMPT COMPARISON
==================================================
Metric         Before    After        Δ
----------------------------------------
Errors              5        0       -5
Warnings            0        0        0
Infos               0        0        0
==================================================

PS C:\Users\Harman\<project_path>\LLM Validator>
```

## Voorbeeld van feedback-loop die (gedeeltelijk) werkt

De code die initieel gegenereerd werd, bevatte 14 errors en 1 warning. Na het toepasses van de feedback-loop, resteren er nog `maar' 5 errors en 0 warnings. De meeste errors zijn na 1 feedback-loop opgelost.

- model: `qwen3:8b` using `ollama serve`
- validator: `docker run -it --rm -p 8888:8888 ghcr.io/validator/validator:latest`
- run: `python loop.py --local`
- reprompt logica:

    ```python
    	reprompt = (
    		f"The following HTML was originally generated for this request:\n"
    		f"{original_prompt}\n\n"
    		f"Here is the generated HTML:\n"
    		f"{html_content}\n\n"
    		f"The W3C HTML validator reported these issues:\n"
    		f"{issues_text}\n\n"
    		f"Please fix every issue listed above and return only the corrected, "
    		f"complete HTML document. Do not include any explanation or markdown fences."
    	)
    ```

> [!NOTE]
> Twee simpele trial-runs met `gemma3:1b` toonde geen verbetering, maar de test met bovenstaande resultaten gebruikte `qwen3:8b`. Ik kan nog niet met zekerheid zeggen waardoor dit verschil komt, wegens het beperkte aantal runs.

Je kan de volledige output hieronder bekijken, of in dit bestand: [./logs/Partial.txt](./logs/Partial.txt)

```powershell
PS C:\Users\Harman\<project_path>\LLM Validator> python loop.py --local

==================================================
STEP 1 — Generating initial HTML
==================================================


HTML saved to ./html/reprompt/generated_2026-03-17_15-58-15.html

==================================================
STEP 2 — Validating initial HTML
==================================================

Validation results saved to ./validation/reprompt/validation_2026-03-17_15-58-15.json
==================================================
Errors:   14
Warnings: 1
Info:     0
==================================================

[ERROR] Line 1, Col 7: Non-space characters found without seeing a doctype first. Expected “<!DOCTYPE html>”.
[ERROR] Line 1, Col 7: Element “head” is missing a required instance of child element “title”.
[ERROR] Line 2, Col 15: Stray doctype.
[ERROR] Line 3, Col 6: Stray start tag “html”.
[ERROR] Line 4, Col 6: Stray start tag “head”.
[ERROR] Line 5, Col 9: Element “title” not allowed as child of element “body” in this context. (Suppressing further errors from this subtree.)
[ERROR] Line 6, Col 24: Attribute “charset” not allowed on element “meta” at this point.
[ERROR] Line 6, Col 24: Element “meta” is missing one or more of the following attributes: “content”, “itemprop”, “property”.
[ERROR] Line 7, Col 72: Attribute “name” not allowed on element “meta” at this point.
[ERROR] Line 7, Col 72: Element “meta” is missing one or more of the following attributes: “itemprop”, “property”.
[ERROR] Line 8, Col 9: Element “style” not allowed as child of element “body” in this context. (Suppressing further errors from this subtree.)
[ERROR] Line 57, Col 7: Stray end tag “head”.
[ERROR] Line 58, Col 6: Start tag “body” seen but an element of the same type was already open.
[ERROR] Line 101, Col 3: Non-space character in page trailer.
[WARNING] Line 1, Col 7: This document appears to be written in English. Consider adding “lang="en"” (or variant) to the “html” start tag.

==================================================
STEP 3 — Re-generating HTML using validation feedback
==================================================


HTML saved to ./html/reprompt/reprompted_2026-03-17_15-58-15.html

==================================================
STEP 4 — Validating reprompted HTML
==================================================

Validation results saved to ./validation/reprompt/validation_reprompted_2026-03-17_15-58-15.json
==================================================
Errors:   5
Warnings: 0
Info:     0
==================================================

[ERROR] Line 1, Col 7: Non-space characters found without seeing a doctype first. Expected “<!DOCTYPE html>”.
[ERROR] Line 1, Col 7: Element “head” is missing a required instance of child element “title”.
[ERROR] Line 2, Col 15: Stray doctype.
[ERROR] Line 3, Col 16: Stray start tag “html”.
[ERROR] Line 3, Col 16: Cannot recover after last error. Any further errors will be ignored.

==================================================
REPROMPT COMPARISON
==================================================
Metric         Before    After        Δ
----------------------------------------
Errors             14        5       -9
Warnings            1        0       -1
Infos               0        0        0
==================================================

PS C:\Users\Harman\<project_path>\LLM Validator>
```

## Voorbeeld van feedback-loop, met meerdere iteraties, die volledig werkt

Je kan de volledige output hieronder bekijken, of in dit bestand: [./logs/Full-loop.txt](./logs/Full-loop.txt)

```powershell
PS C:\Users\Harman\<project_path>\LLM Validator> python main.py --local --validate-and-regenerate 3

==================================================
STEP 1 — Validating existing HTML
==================================================

Validation results saved to ./validation/validation_2026-04-07_20-10-22.json
==================================================
Errors:   5
Warnings: 0
Info:     0
==================================================

[ERROR] Line 1, Col 142: Non-space characters found without seeing a doctype first. Expected “<!DOCTYPE html>”.
[ERROR] Line 1, Col 142: Element “head” is missing a required instance of child element “title”.
[ERROR] Line 4, Col 15: Stray doctype.
[ERROR] Line 5, Col 16: Stray start tag “html”.
[ERROR] Line 5, Col 16: Cannot recover after last error. Any further errors will be ignored.

==================================================
ITERATION 1/3 — Re-generating HTML using validation feedback
==================================================


HTML saved to ./html/reprompt/reprompted_2026-04-07_20-10-22_iter1.html

==================================================
ITERATION 1/3 — Validating reprompted HTML
==================================================

Validation results saved to ./validation/reprompt/validation_reprompted_2026-04-07_20-10-22_iter1.json
==================================================
Errors:   5
Warnings: 0
Info:     0
==================================================

[ERROR] Line 1, Col 7: Non-space characters found without seeing a doctype first. Expected “<!DOCTYPE html>”.
[ERROR] Line 1, Col 7: Element “head” is missing a required instance of child element “title”.
[ERROR] Line 2, Col 15: Stray doctype.
[ERROR] Line 3, Col 16: Stray start tag “html”.
[ERROR] Line 3, Col 16: Cannot recover after last error. Any further errors will be ignored.

==================================================
ITERATION 2/3 — Re-generating HTML using validation feedback
==================================================


HTML saved to ./html/reprompt/reprompted_2026-04-07_20-10-22_iter2.html

==================================================
ITERATION 2/3 — Validating reprompted HTML
==================================================

Validation results saved to ./validation/reprompt/validation_reprompted_2026-04-07_20-10-22_iter2.json
==================================================
Errors:   0
Warnings: 0
Info:     0
==================================================


==================================================
ITERATION 3/3 — Re-generating HTML using validation feedback
==================================================


HTML saved to ./html/reprompt/reprompted_2026-04-07_20-10-22_iter3.html

==================================================
ITERATION 3/3 — Validating reprompted HTML
==================================================

Validation results saved to ./validation/reprompt/validation_reprompted_2026-04-07_20-10-22_iter3.json
==================================================
Errors:   0
Warnings: 0
Info:     0
==================================================


==================================================
REPROMPT COMPARISON (after 3 iteration(s))
==================================================

Metric         Before    After        Δ
----------------------------------------
Errors              5        0       -5
Warnings            0        0        0
Infos               0        0        0
==================================================
```
