"""The forensic trust layer: data-integrity checks on cited studies.

Lifted from paperscope (MIT). These guard the checks that let a clinical answer
flag a source whose reported statistics cannot be real before relying on it.
"""
import pytest

pytest.importorskip("scipy")  # forensic recomputation checks need scipy

from localevidence import forensic as F


def test_grim_detects_impossible_mean():
    # 5.19 * 28 = 145.32 — not achievable as sum/n for integer-scored data
    assert F.grim("5.19", 28)["possible"] is False


def test_grim_accepts_possible_mean():
    assert F.grim("5.18", 28)["possible"] is True  # 5.18 * 28 = 145.04 ~ 145


def test_debit_runs():
    assert isinstance(F.debit(88.5, 26, dp=1)["possible"], bool)


def test_correlation_bound_flags_impossible_r():
    # pre/post/change SDs implying |r| > 1
    result = F.correlation_bound(0.13, 0.27, 0.03)
    assert result["detail"].startswith("FAIL")


def test_grimmer_and_benford_importable_and_run():
    assert isinstance(F.grimmer("5.18", "1.20", 28)["possible"], bool)
    out = F.benfords_law([123, 234, 345, 456, 567, 678, 789, 891, 912, 145, 267, 389])
    assert "detail" in out
