from experiments.run_batch import derive_seed, load_completed_run_ids, safe_name
from util.results_store import append_run_record, make_run_record


def test_derive_seed_is_deterministic():
    assert derive_seed(1000, "modelA", "p1", 1) == derive_seed(1000, "modelA", "p1", 1)


def test_derive_seed_varies_by_trial():
    assert derive_seed(1000, "modelA", "p1", 1) != derive_seed(1000, "modelA", "p1", 2)


def test_derive_seed_varies_by_prompt():
    assert derive_seed(1000, "modelA", "p1", 1) != derive_seed(1000, "modelA", "p2", 1)


def test_safe_name_strips_colon():
    assert safe_name("qwen3:8b") == "qwen3-8b"


def _record(**overrides):
    base = dict(
        run_id="r1",
        timestamp="t",
        model="m",
        prompt_id="p1",
        difficulty="simple",
        condition="initial",
        trial=1,
        iteration=0,
        temperature=0.2,
        seed=1,
        errors=2,
        warnings=0,
        infos=0,
        error_categories={},
        gen_time_s=1.0,
        prompt_eval_count=1,
        eval_count=1,
        was_cleaned=False,
        html_path="a.html",
        validation_path="a.json",
    )
    base.update(overrides)
    return make_run_record(**base)


def test_load_completed_run_ids_missing_csv_returns_empty(tmp_path):
    path = str(tmp_path / "results.csv")
    assert load_completed_run_ids(path, ["feedback", "blind"]) == set()


def test_load_completed_run_ids_already_perfect_initial_counts_as_done(tmp_path):
    path = str(tmp_path / "results.csv")
    append_run_record(path, _record(run_id="r1", errors=0, warnings=0, infos=0))

    assert load_completed_run_ids(path, ["feedback", "blind"]) == {"r1"}


def test_load_completed_run_ids_missing_condition_is_not_done(tmp_path):
    path = str(tmp_path / "results.csv")
    append_run_record(path, _record(run_id="r1", condition="initial"))
    append_run_record(path, _record(run_id="r1", condition="feedback", iteration=1))

    assert load_completed_run_ids(path, ["feedback", "blind"]) == set()


def test_load_completed_run_ids_all_conditions_present_is_done(tmp_path):
    path = str(tmp_path / "results.csv")
    append_run_record(path, _record(run_id="r1", condition="initial"))
    append_run_record(path, _record(run_id="r1", condition="feedback", iteration=1))
    append_run_record(path, _record(run_id="r1", condition="blind", iteration=1))

    assert load_completed_run_ids(path, ["feedback", "blind"]) == {"r1"}


def test_load_completed_run_ids_tracks_each_run_id_independently(tmp_path):
    path = str(tmp_path / "results.csv")
    append_run_record(path, _record(run_id="done", errors=0, warnings=0, infos=0))
    append_run_record(path, _record(run_id="partial", condition="initial"))
    append_run_record(
        path, _record(run_id="partial", condition="feedback", iteration=1)
    )

    assert load_completed_run_ids(path, ["feedback", "blind"]) == {"done"}


def test_load_completed_run_ids_no_initial_row_is_not_done(tmp_path):
    path = str(tmp_path / "results.csv")
    append_run_record(path, _record(run_id="r1", condition="feedback", iteration=1))
    append_run_record(path, _record(run_id="r1", condition="blind", iteration=1))

    assert load_completed_run_ids(path, ["feedback", "blind"]) == set()
