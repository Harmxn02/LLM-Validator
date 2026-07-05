from experiments.run_batch import derive_seed, safe_name


def test_derive_seed_is_deterministic():
    assert derive_seed(1000, "modelA", "p1", 1) == derive_seed(1000, "modelA", "p1", 1)


def test_derive_seed_varies_by_trial():
    assert derive_seed(1000, "modelA", "p1", 1) != derive_seed(1000, "modelA", "p1", 2)


def test_derive_seed_varies_by_prompt():
    assert derive_seed(1000, "modelA", "p1", 1) != derive_seed(1000, "modelA", "p2", 1)


def test_safe_name_strips_colon():
    assert safe_name("qwen3:8b") == "qwen3-8b"
