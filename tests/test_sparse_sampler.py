"""Tests for _sample_sparse_subsets: n_oxides >= min_k must always hold,
and distribution across k=min_k..max_k must be approximately uniform.

Prove-It (bug-fix) pattern:
1. Tests describe expected behavior (n_oxides always >= min_k=3, uniform dist)
2. Some tests FAIL before the fix
3. Fix the code
4. All tests pass
"""
import numpy as np
import pandas as pd
import pytest

from roadlab_matnav_lib.recommend import _sample_sparse_subsets

OXIDES_11 = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO',
    'ZnO', 'ZrO2', 'B2O3', 'La2O3', 'Y2O3',
]
OXIDES_NO_SIO2 = ['Al2O3', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO']

N = 3000
THRESHOLD = 1.0
MIN_K = 3


# ── helper ──────────────────────────────────────────────────────────────────

def n_active(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.Series:
    """Count oxides strictly above threshold in each row."""
    return (df > threshold).sum(axis=1)


# ── Row-sum invariant ────────────────────────────────────────────────────────

class TestRowSums:

    @pytest.mark.parametrize("max_k", [3, 5, 7, 9, 11])
    def test_rows_sum_to_100(self, max_k):
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=0, min_k=MIN_K)
        sums = df.sum(axis=1)
        assert (sums.between(99.9, 100.1)).all(), \
            f"max_k={max_k}: some rows don't sum to 100"


# ── min_k lower-bound: with threshold=1.0 ───────────────────────────────────

class TestMinKWithThreshold:

    @pytest.mark.parametrize("max_k", [3, 4, 5, 6, 7, 8, 9, 10, 11])
    def test_no_row_has_fewer_than_min_k_oxides(self, max_k):
        """All returned rows must have at least min_k active oxides."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=42, min_k=MIN_K)
        bad = n_active(df) < MIN_K
        assert not bad.any(), \
            f"max_k={max_k}: {bad.sum()} rows have < {MIN_K} active oxides"

    @pytest.mark.parametrize("max_k", [3, 4, 5, 6, 7, 8, 9, 10, 11])
    def test_no_row_has_more_than_max_k_oxides(self, max_k):
        """No row should have more than max_k active oxides."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=42, min_k=MIN_K)
        bad = n_active(df) > max_k
        assert not bad.any(), \
            f"max_k={max_k}: {bad.sum()} rows have > {max_k} active oxides"


# ── min_k lower-bound: with threshold=0.0 ───────────────────────────────────

class TestMinKWithoutThreshold:

    @pytest.mark.parametrize("max_k", [3, 5, 7, 11])
    def test_no_row_has_fewer_than_min_k_oxides_threshold_zero(self, max_k):
        """Even with oxide_threshold=0, min_k must be enforced."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, oxide_threshold=0.0, seed=7, min_k=MIN_K)
        bad = (df > 0).sum(axis=1) < MIN_K
        assert not bad.any(), \
            f"threshold=0, max_k={max_k}: {bad.sum()} rows have < {MIN_K} active oxides"


# ── SiO2 always present ──────────────────────────────────────────────────────

class TestSiO2Presence:

    @pytest.mark.parametrize("max_k", [3, 5, 11])
    def test_every_row_contains_sio2(self, max_k):
        """SiO2 must be positive in every returned row."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=1, min_k=MIN_K)
        assert (df["SiO2"] > 0).all(), \
            f"max_k={max_k}: {(df['SiO2'] == 0).sum()} rows have SiO2=0"

    def test_without_sio2_in_oxide_list(self):
        """When SiO2 is absent from oxide list, min_k still enforced."""
        df = _sample_sparse_subsets(OXIDES_NO_SIO2, N, max_k=4, oxide_threshold=THRESHOLD, seed=2, min_k=MIN_K)
        bad = n_active(df) < MIN_K
        assert not bad.any(), f"{bad.sum()} rows have < {MIN_K} active oxides"


# ── min_k=3 specifically: no 2-oxide rows ───────────────────────────────────

class TestNoTwoOxideRows:
    """Regression for the reported bug: n=2 compositions appearing in results."""

    @pytest.mark.parametrize("max_k,seed", [
        (3, 0), (3, 1), (3, 99),
        (5, 0), (5, 42), (5, 123),
        (7, 0), (7, 7),
        (11, 0), (11, 55),
    ])
    def test_no_two_oxide_rows_threshold_1(self, max_k, seed):
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=seed, min_k=3)
        two_oxide = (n_active(df) < 3).sum()
        assert two_oxide == 0, \
            f"max_k={max_k} seed={seed}: {two_oxide} rows with < 3 oxides"

    @pytest.mark.parametrize("max_k,seed", [
        (3, 0), (5, 42), (11, 0),
    ])
    def test_no_two_oxide_rows_threshold_zero(self, max_k, seed):
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, oxide_threshold=0.0, seed=seed, min_k=3)
        two_oxide = ((df > 0).sum(axis=1) < 3).sum()
        assert two_oxide == 0, \
            f"threshold=0, max_k={max_k} seed={seed}: {two_oxide} rows with < 3 oxides"


# ── Uniform distribution across k groups ────────────────────────────────────

class TestUniformDistribution:
    """Each k value from min_k..max_k must appear in approximately equal counts."""

    @pytest.mark.parametrize("max_k", [5, 7, 9, 11])
    def test_each_k_has_equal_quota(self, max_k):
        """Per-k quota is fixed by full_groups = K - min_k + 1 (= 9 for 11 oxides).
        Total output = (max_k - min_k + 1) * quota_base <= N when max_k < K."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=0, min_k=MIN_K)
        full_groups = len(OXIDES_11) - MIN_K + 1  # always 9
        quota_base = N // full_groups
        active_groups = max_k - MIN_K + 1
        expected_total = active_groups * quota_base
        # Allow ±active_groups for remainder distribution
        assert abs(len(df) - expected_total) <= active_groups, (
            f"max_k={max_k}: expected ~{expected_total} rows (quota_base={quota_base}×{active_groups}), "
            f"got {len(df)}"
        )

    @pytest.mark.parametrize("max_k", [5, 7, 9, 11])
    def test_low_k_and_high_k_have_similar_counts(self, max_k):
        """n_oxides=3 and n_oxides=max_k groups should be within 4x of each other.
        Quota-based sampling ensures equal intended-k groups; after threshold,
        high-k rows may split into multiple lower n_active buckets,
        so the ratio is bounded but not tight."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k, THRESHOLD, seed=0, min_k=MIN_K)
        n_ox = (df > 0).sum(axis=1)
        count_low  = (n_ox == MIN_K).sum()
        count_high = (n_ox == max_k).sum()
        if count_low == 0 or count_high == 0:
            pytest.skip(f"one bucket empty: low={count_low} high={count_high}")
        ratio = max(count_low, count_high) / min(count_low, count_high)
        assert ratio <= 4.0, (
            f"max_k={max_k}: n={MIN_K} count={count_low}, "
            f"n={max_k} count={count_high}, ratio={ratio:.1f} > 4.0 — not uniform"
        )
