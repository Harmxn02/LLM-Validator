# LLM-Validator

Research pipeline for the proposal *"Towards Perfect Code Generation: Iterative
API-based Feedback Loops"*. It evaluates whether an iterative
validation-and-regeneration feedback loop improves the W3C-defined markup
correctness of LLM-generated HTML, and whether smaller local models can
close the gap to larger ones through that loop.

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

### Design principles

- **Fixed prompt set.** Every model is evaluated on the exact same, ordered
  set of prompts (`prompts/prompts.json`, 51 prompts across simple/medium/
  difficult tiers via `iter_prompts()`) — no per-run random sampling, so
  comparisons across models are apples-to-apples.
- **Clean generation output.** A system prompt instructs raw-HTML-only
  output, and `clean_html_output()` defensively strips any markdown fences
  or explanatory preamble the model adds anyway, before validation — so
  formatting slip-ups are never counted as HTML errors.
- **Reproducible runs.** Every run logs its `temperature` and a
  deterministically-derived `seed` (`experiments/run_batch.py::derive_seed`),
  so a given config always reproduces the same run matrix.
- **One validator.** All validation goes through a single local W3C Docker
  validator (`util/config.py`); `analysis/check_validator_determinism.py`
  exists to confirm that validator itself returns consistent results before
  trusting any before/after delta.
- **Repeated trials.** Each (model, prompt) combination runs `trials` times
  (config-driven) so variance and confidence intervals are computable
  instead of relying on single-run anecdotes.
- **Blind-ablation control.** For a given initial generation, both a
  `feedback` reprompt (sees the validator's errors) and a `blind` reprompt
  (told only to "review and fix") are run, so the analysis can isolate how
  much of any improvement is attributable to the validator feedback itself.
- **Structured results.** Every iteration of every run is appended as one
  row to `results/experiments.csv` (`util/results_store.py`) — errors,
  warnings, infos, an error-category breakdown, generation time, and token
  counts — instead of one-off JSON files with no aggregation.

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
  config.py                    Local validator URL (single source of truth)
experiments/run_batch.py      Sweeps model x prompt x trial x condition, logs every iteration to results/experiments.csv
analysis/
  stats.py                     Mean +/- 95% CI, paired Wilcoxon tests (before/after, feedback vs blind),
                                error-category breakdown, iteration-cutoff (quality vs. cost) analysis
  plots.py                     Convergence trajectories, per-model boxplots, cost/quality tradeoff scatter
  check_validator_determinism.py  Re-validates the same file N times to rule out validator noise
main.py                       Lightweight single-run CLI for manual/ad hoc checks (not used for reported results)
tests/                        pytest unit tests for the parsing/prompt/results logic
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

## Scope

**RQ1** is evaluated as: does the feedback loop reduce W3C markup-validity
errors/warnings (doctype, tag nesting, required attributes, etc.) —
"code correctness" is defined operationally as validator conformance.

**RQ2** is evaluated as smaller vs. larger *local* Ollama models, using
generation latency and token counts as the cost proxy — this pipeline does
not currently include a cloud/API-based model baseline.
