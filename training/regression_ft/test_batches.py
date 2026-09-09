import math
import random

from batches import make_batches, learning_rate


def test_exact_once_across_short_long_and_tail():
    rng = random.Random(1)
    rows = [{'duration': rng.uniform(0.1, 45)} for _ in range(10003)]
    rows += [{'duration': 170}, {'duration': 0.01}]
    batches = make_batches(rows, 22)
    assert batches == make_batches(rows, 22)
    assert batches != make_batches(rows, 23)
    flattened = [i for b in batches for i in b]
    assert sorted(flattened) == list(range(len(rows)))
    for b in batches:
        assert len(b) <= 16
        assert len(b) == 1 or max(rows[i]['duration'] for i in b) * len(b) <= 120
    assert any(10003 in b for b in batches)


def test_schedule_reaches_peak_and_end():
    rates = [learning_rate(i, 4001, 1e-6, 1e-7, .03) for i in range(1, 4002)]
    assert math.isclose(max(rates), 1e-6)
    assert math.isclose(rates[-1], 1e-7)
    assert all(0 < x <= 1e-6 for x in rates)
    peak = rates.index(max(rates))
    assert all(a <= b for a, b in zip(rates[:peak], rates[1:peak + 1]))
    assert all(a >= b for a, b in zip(rates[peak:], rates[peak + 1:]))
