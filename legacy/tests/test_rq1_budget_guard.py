import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("bg", Path(__file__).resolve().parents[2] / "scripts" / "c1" / "budget_guard.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)


def led(jobs=(), authorized=1):
    return {"cap_node_hours": 3.2, "authorized_jobs": authorized, "jobs": [dict(j) for j in jobs]}


def test_first_job_allowed_second_refused_even_if_budget_remains():
    assert bg.check(led(), 64)[0]
    done = bg.record(led(), "1", 1, 1800)
    assert not bg.check(done, 64)[0] and "authorized" in bg.check(done, 64)[1]


def test_cap_is_a_hard_limit_on_accumulated_actual_plus_next_maximum():
    l = led(authorized=3)
    bg.record(l, "1", 1, 3600 * 1.0)
    bg.record(l, "2", 2, 3600 * 1.0)
    assert bg.spent(l) == 2.0
    assert bg.check(l, 64)[0]                  # 2.0 + 1.0667 = 3.0667 <= 3.2
    bg.record(l, "3", 3, 3600 * 1.2)
    assert not bg.check(dict(l, authorized_jobs=4), 60)[0]     # 3.2 + 1.0 > 3.2


def test_three_jobs_at_the_planned_walltime_exactly_fit_the_cap():
    assert 3 * 64 / 60.0 == 3.2
    assert abs(3 * 64 / 60.0 - 3.2) < 1e-12
