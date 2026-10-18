import isolation_targets as targets

from musicset.parallel import map_isolated, run_isolated, worker_count


def test_result_comes_back():
    outcome = run_isolated(targets.square, 3)
    assert outcome.ok and outcome.value == 9


def test_exception_is_reported_not_retried():
    outcome = run_isolated(targets.fail, 0, retries=3)
    assert not outcome.ok and outcome.error == "ValueError: bad score"


def test_native_crash_costs_only_the_task():
    outcome = run_isolated(targets.crash, 0)
    assert not outcome.ok and outcome.error is not None
    assert outcome.error.startswith("crashed")


def test_hang_is_killed():
    outcome = run_isolated(targets.hang, 0, timeout=1)
    assert not outcome.ok and outcome.error == "timed out after 1s"


def test_map_survives_a_crashing_neighbour():
    results = dict(map_isolated(targets.square, [1, 2, 3], workers=2))
    assert {k: v.value for k, v in results.items()} == {1: 1, 2: 4, 3: 9}


def test_worker_count_env_override(monkeypatch):
    monkeypatch.setenv("MUSICSET_WORKERS", "3")
    assert worker_count() == 3
    monkeypatch.setenv("MUSICSET_WORKERS", "zero")
    assert worker_count() >= 1
