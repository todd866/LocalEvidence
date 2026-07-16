"""acquire() must not hang forever on a slow mirror.

Two bounds, both env-tunable:
- LE_PULL_BUDGET_S: per-pull timeout; a single slow pull is abandoned and
  recorded as a failure instead of blocking the whole run.
- LE_ACQUIRE_BUDGET_S: per-question acquisition budget; once spent, remaining
  candidates are skipped rather than pulled.
"""
from __future__ import annotations

import os
import time
import unittest
from unittest import mock


class PullBoundedTest(unittest.TestCase):
    def test_returns_a_fast_result(self):
        from localevidence.acquire import _pull_bounded
        self.assertEqual(_pull_bounded(lambda: {"_status": "pulled"}, timeout_s=1.0),
                         {"_status": "pulled"})

    def test_times_out_a_slow_pull_promptly(self):
        from localevidence.acquire import _pull_bounded, PullTimeout
        started = time.monotonic()
        with self.assertRaises(PullTimeout):
            _pull_bounded(lambda: time.sleep(0.6), timeout_s=0.05)
        self.assertLess(time.monotonic() - started, 0.4,
                        "should return well before the slow pull finishes")

    def test_zero_timeout_disables_the_bound(self):
        from localevidence.acquire import _pull_bounded
        self.assertEqual(_pull_bounded(lambda: 42, timeout_s=0), 42)

    def test_propagates_the_pull_exception(self):
        from localevidence.acquire import _pull_bounded

        def boom():
            raise ValueError("mirror said no")

        with self.assertRaises(ValueError):
            _pull_bounded(boom, timeout_s=1.0)


def _candidates(n):
    from localevidence.discovery import Candidate
    return [Candidate(doi=f"10.1/{i}", title=f"Paper {i}", tier="rct") for i in range(n)]


def _triage(n):
    from localevidence.triage import TriageResult
    cands = _candidates(n)
    return TriageResult(ranked=cands, to_acquire=cands, in_library=[])


class AcquireBoundsTest(unittest.TestCase):
    def test_slow_pull_is_recorded_as_failure_not_a_hang(self):
        from localevidence import acquire as acq

        def slow_pull(*a, **k):
            time.sleep(0.6)
            return {"_status": "pulled"}

        with mock.patch("localevidence.library.pull", slow_pull), \
             mock.patch.dict(os.environ, {"LE_PULL_BUDGET_S": "0.05"}):
            started = time.monotonic()
            report = acq.acquire(_triage(1), pace_s=0, verbose=False)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(report.pulled, 0)
        self.assertEqual(len(report.failures), 1)

    def test_acquisition_budget_stops_further_pulls(self):
        from localevidence import acquire as acq
        calls = {"n": 0}

        def paced_pull(*a, **k):
            calls["n"] += 1
            time.sleep(0.1)
            return {"_status": "pulled", "text": "x", "title": "t", "doi": "d"}

        with mock.patch("localevidence.library.pull", paced_pull), \
             mock.patch.dict(os.environ, {"LE_ACQUIRE_BUDGET_S": "0.05",
                                          "LE_PULL_BUDGET_S": "0"}):
            report = acq.acquire(_triage(3), pace_s=0, verbose=False)
        # first pull runs and spends the budget; the other two are skipped
        self.assertEqual(calls["n"], 1)
        self.assertEqual(report.budget_skipped, 2)

    def test_acquisition_budget_bounds_even_a_single_slow_pull(self):
        # Even with the per-pull timeout disabled, the acquisition budget is a
        # hard wall-clock cap: a single slow pull is bounded by the time left.
        from localevidence import acquire as acq

        def slow_pull(*a, **k):
            time.sleep(0.6)
            return {"_status": "pulled"}

        with mock.patch("localevidence.library.pull", slow_pull), \
             mock.patch.dict(os.environ, {"LE_ACQUIRE_BUDGET_S": "0.1",
                                          "LE_PULL_BUDGET_S": "0"}):
            started = time.monotonic()
            report = acq.acquire(_triage(1), pace_s=0, verbose=False)
        self.assertLess(time.monotonic() - started, 0.4,
                        "the acquisition budget must cap a single pull too")
        self.assertEqual(report.pulled, 0)


if __name__ == "__main__":
    unittest.main()
