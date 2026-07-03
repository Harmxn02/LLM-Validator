# Technical risks

Supporting material for the proposal's risk-analysis section (peer-review
comment #9). These are risks specific to this pipeline's implementation, not
a general project-management risk list.

## Reproducibility

- **Local model non-determinism.** Ollama's `seed`/`temperature` options
  reduce but do not guarantee bit-exact reproducibility across different
  hardware, driver versions, or Ollama releases. Mitigation: report
  mean/CI across `trials` repeats rather than trusting any single run;
  don't compare absolute values across machines.
- **Validator version drift.** `util/config.py` points at a local Docker
  validator run from the `ghcr.io/validator/validator:latest` tag. Re-running
  the experiment weeks apart could silently pull an updated validator with
  different rule behavior, breaking comparability across sessions.
  Mitigation: pin to a specific image digest before running the final
  reported experiment, and record it in the write-up.
- **Non-idempotent batch runs.** `experiments/run_batch.py` appends to
  `results/experiments.csv` on every invocation; if a run crashes partway
  (Ollama or the validator container going down mid-sweep) and is restarted
  from scratch, already-logged runs will be duplicated rather than resumed.
  Mitigation: treat a `results/experiments.csv` as belonging to one
  from-scratch batch invocation; don't re-run without clearing it first.

## Measurement validity

- **Validator noise.** `analysis/check_validator_determinism.py` exists
  specifically to rule out the validator itself as a source of noise before
  trusting any before/after delta — run it once against a representative
  file before treating results as final.
- **Heuristic output cleaning.** `clean_html_output()` strips markdown
  fences/preamble from model output before validation. This is a heuristic:
  in principle a generation that legitimately contains a fenced code sample
  (e.g. a `<pre>` block showing HTML-in-HTML) could be mis-trimmed. Every
  run logs `was_cleaned`, so flagged runs can be spot-checked.
- **Statistical power.** With only `trials` repeats per (model, prompt,
  condition), the Wilcoxon signed-rank test has a floor on the smallest
  attainable p-value (e.g. ~0.06 at n=5) — a real effect can fail to reach
  significance purely from sample size, not because the effect is absent.
  Mitigation: `iteration_cutoff_analysis` and the CI widths in
  `analysis/stats.py` make it possible to see whether non-significance is
  "no effect" or "not enough trials"; increase `trials` in
  `config/experiment.json` if CIs are too wide to draw conclusions.

## Scope

- **RQ1 is scoped to W3C markup-validity**, not semantic-tag appropriateness
  (`<div>` vs `<section>`/`<nav>`/`<article>`). Conclusions about "code
  correctness" should be read as "W3C-validator-defined correctness," which
  is a narrower claim than the original proposal's framing around
  accessibility/semantic best practices.
- **RQ2 has no cloud/API-model baseline.** Findings compare smaller vs.
  larger *local* Ollama models only; they do not establish whether local
  models with the feedback loop can match commercial cloud models, and
  generation-time/token-count cost proxies do not translate directly to
  real deployment costs (API pricing, hosting infrastructure).
