from util.results_store import append_run_record, load_results, make_run_record


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
        warnings=1,
        infos=0,
        error_categories={"stray-tag": 2},
        gen_time_s=1.5,
        prompt_eval_count=10,
        eval_count=20,
        was_cleaned=True,
        html_path="a.html",
        validation_path="a.json",
    )
    base.update(overrides)
    return make_run_record(**base)


def test_round_trip_preserves_values(tmp_path):
    path = str(tmp_path / "results.csv")
    append_run_record(path, _record())

    df = load_results(path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["model"] == "m"
    assert row["errors"] == 2
    assert row["error_categories"] == {"stray-tag": 2}


def test_append_writes_header_only_once(tmp_path):
    path = str(tmp_path / "results.csv")
    for i in range(3):
        append_run_record(path, _record(run_id=f"r{i}"))

    df = load_results(path)

    assert len(df) == 3
    assert list(df["run_id"]) == ["r0", "r1", "r2"]
