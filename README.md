# LLM-Validator

Research pipeline for the proposal *"Towards Perfect Code Generation: Iterative
API-based Feedback Loops"*: can a validation-and-regeneration feedback loop
improve LLM-generated HTML, and does it let smaller/cheaper models close the
gap to larger ones?

> The original exploratory single-run version of this project (and its
> README) has been moved to [`deprecated/`](deprecated/README.md). Everything
> below describes the current, batch-driven pipeline.

## How it works

```
                         ┌────────────────────────────┐
                         │   config/experiment.json    │
                         │ models, prompts, trials,    │
                         │ iterations, temperature     │
                         └──────────────┬───────────────┘
                                        ▼
prompts/prompts.json ──▶ experiments/run_batch.py ──▶ results/experiments.csv
  {id, difficulty,          for each (model, prompt,        one row per
   text}                     trial):                        (run, iteration)
                             1. generate initial HTML
                                (util/generation.py)
                             2. validate it (util/validation.py
                                against a local W3C Docker validator)
                             3. reprompt loop, twice in parallel:
                                - "feedback": model sees the validator's
                                  errors (the actual guardrail)
                                - "blind": model just told to "review and
                                  fix" (ablation control — isolates how much
                                  of any improvement is the validator
                                  feedback itself vs. a second attempt)
                                stopping early once a run hits 0 issues
                                        │
                                        ▼
                         analysis/stats.py, analysis/plots.py
                         significance tests, CIs, convergence plots
```

### Why a batch pipeline instead of one-off runs

The original approach ran the loop manually, once, per model, on a randomly
chosen prompt — fine for a demo, not for evidence in a research write-up.
The batch pipeline exists to remove several confounds from that approach:

| Confound in the manual flow | Fix |
| --- | --- |
| Different models/runs saw different random prompts | `iter_prompts()` gives every model the exact same fixed, ordered prompt set |
| Model output sometimes wrapped in ` ```html ` fences or had explanatory text, which the validator would flag as errors unrelated to actual HTML quality | `clean_html_output()` strips fences/preamble before validation; a system prompt also instructs raw-HTML-only output |
| No record of temperature/seed — results not reproducible | Every run logs `temperature` and a deterministically-derived `seed` |
| `--local` flag let cloud and local validators be mixed across a comparison | Pinned to a single local Docker validator (`util/config.py`) |
| n=1 anecdotes ("this model seems to work better") | `trials` in the config repeats each (model, prompt) combination N times so variance/CIs and significance tests are possible |
| No control condition — impossible to tell if the validator feedback matters vs. just a second generation attempt | `blind` condition: reprompts without the error list, paired with `feedback` on the same initial HTML |
| Results scattered across timestamped JSON files, no aggregation | `util/results_store.py` appends one structured CSV row per run/iteration |

## Repository layout

```
config/experiment.json        Batch sweep config: models, prompts, trials, iterations, temperature, seed
prompts/prompts.json          {id, difficulty, text} — the fixed prompt set (51 prompts, 3 difficulty tiers)
util/
  generation.py                Ollama call, HTML-only system prompt, fence/preamble stripping, timing/token capture
  validation.py                W3C validator call, error/warning/info parsing, error-category taxonomy
  prompts.py                   Prompt loading/selection, feedback + blind reprompt construction
  pipeline.py                  generate → validate → reprompt loop, with a per-iteration logging callback
  results_store.py             Structured run-record schema + CSV append/load (pandas)
  config.py                    Local validator URL (single source of truth, no cloud/local mixing)
experiments/run_batch.py      Sweeps model x prompt x trial x condition, logs every iteration to results/experiments.csv
analysis/
  stats.py                     Mean +/- 95% CI, paired Wilcoxon tests (before/after, feedback vs blind), error-category breakdown
  plots.py                     Convergence trajectories, per-model boxplots, cost/quality tradeoff scatter
  check_validator_determinism.py  Re-validates the same file N times to rule out validator noise
main.py                       Lightweight single-run CLI for manual/ad hoc checks (not used for reported results)
tests/                        pytest unit tests for the parsing/prompt/results logic
deprecated/                   Original README from the manual, single-run exploration phase
```

## Running it

Prerequisites: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/)
running locally, and a local W3C validator container.

```bash
uv sync

ollama serve                                                          # terminal 1
docker run -p 8888:8888 ghcr.io/validator/validator:latest --port 8888  # terminal 2

# terminal 3
uv run pytest                                                          # sanity-check the pipeline first
uv run python -m analysis.check_validator_determinism --html-file some.html  # optional: confirm the validator is stable

# edit config/experiment.json (models, trials, iterations) for your run, then:
uv run python -m experiments.run_batch                                # generates results/experiments.csv
uv run python -m analysis.stats --results-csv results/experiments.csv # tables + significance tests
uv run python -m analysis.plots --results-csv results/experiments.csv # writes PNGs to plots/
```

For a single manual run (e.g. to sanity-check a new model by hand), `main.py`
still works and takes the same generation parameters as flags:

```bash
uv run python main.py --model qwen3:8b --temperature 0.2 --seed 42
uv run python main.py --validate-only --html-file path/to.html
uv run python main.py --validate-and-regenerate 3 --html-file path/to.html --blind
```

## Results schema

Every row in `results/experiments.csv` (see `util/results_store.py` for the
authoritative schema) is one iteration of one run:

| Column | Meaning |
| --- | --- |
| `run_id` | Identifies the (model, prompt, trial) triple — shared across `initial`/`feedback`/`blind` rows so they can be paired |
| `condition` | `initial` (before any reprompting), `feedback`, or `blind` |
| `iteration` | 0 for the initial generation, 1..N for reprompt iterations |
| `difficulty` | `simple` / `medium` / `difficult`, from the prompt set |
| `temperature`, `seed` | Generation parameters, for reproducibility |
| `errors`, `warnings`, `infos` | W3C validator counts |
| `error_categories` | JSON dict of coarse error categories (missing-doctype, stray-tag, disallowed-attribute, ...) |
| `gen_time_s`, `prompt_eval_count`, `eval_count` | Generation latency and token counts, for cost/quality tradeoff analysis |
| `was_cleaned` | Whether the raw model output needed fence/preamble stripping |
| `html_path`, `validation_path` | Where the generated HTML and raw validator JSON were saved |

## Known limitations

This pipeline currently measures **W3C markup-validity** (does the HTML
conform to the HTML5 spec: doctype, tag nesting, required attributes) — it
does **not** yet measure the semantic-tag misuse (`<div>` overuse vs.
`<section>`/`<article>`/`<nav>`) that the underlying research proposal is
actually about, since the W3C validator does not flag that as an error. See
the open peer-review discussion for what's being added to close this and
other gaps.
