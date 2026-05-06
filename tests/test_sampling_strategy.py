"""Tests for sampling strategy selection in _sample_sparse_subsets and sample_simplex.

Verifies the user's question:
  "n_oxides=9 일때 10만 샘플 중 9개 이하 필터인가, 아니면 다르게 샘플링하는가?"

Answer: max_n_oxides < len(active_oxides) → sparse subset sampling (max_k 직접 제한)
        max_n_oxides == len(active_oxides) → full simplex sampling (11개 전체)

TDD: write tests → confirm some FAIL → fix → all pass.
"""
import numpy as np
import pandas as pd
import pytest

from roadlab_matnav_lib.recommend import _sample_sparse_subsets, sample_simplex, _apply_threshold

OXIDES_11 = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO',
    'ZnO', 'ZrO2', 'B2O3', 'La2O3', 'Y2O3',
]
N = 10_000
THRESHOLD = 1.0


# ── Full simplex (max_n_oxides=11): nearly all samples survive ───────────────

class TestFullSimplexSurvivalRate:
    """sample_simplex on 11 oxides loses very few rows to threshold=1%."""

    def test_survival_rate_above_99pct(self):
        """Dirichlet(α=1) on 11 oxides: each component averages 9.1% >> 1%,
        so almost no rows are dropped by threshold.  Expect ≥99% survival."""
        raw = sample_simplex(OXIDES_11, N, seed=0)
        after = _apply_threshold(raw, THRESHOLD)
        rate = len(after) / N
        assert rate >= 0.99, f"survival rate {rate:.3f} < 0.99 for full simplex"

    def test_full_simplex_n_oxides_mostly_high(self):
        """After threshold, almost all rows should have n_oxides >= 9."""
        raw = sample_simplex(OXIDES_11, N, seed=0)
        after = _apply_threshold(raw, THRESHOLD)
        n_ox = (after > 0).sum(axis=1)
        frac_high = (n_ox >= 9).sum() / len(after)
        assert frac_high >= 0.90, \
            f"only {frac_high:.2%} of full-simplex rows have n_oxides>=9"

    def test_full_simplex_no_max_k_constraint(self):
        """Full simplex can produce n=11 rows; sparse sampler with max_k<11 cannot."""
        raw = sample_simplex(OXIDES_11, N, seed=0)
        after = _apply_threshold(raw, THRESHOLD)
        n_ox = (after > 0).sum(axis=1)
        assert (n_ox == 11).any(), "full simplex should produce some n=11 rows"


# ── Sparse sampler (max_n_oxides=9): directly limits max k, rows dropped ─────

class TestSparseSamplerDropRate:
    """_sample_sparse_subsets loses rows to threshold+SiO2 filter."""

    def test_sparse_max9_returns_fewer_than_n_samples(self):
        """With full_groups normalization, max_k=9 produces 7/9 * N rows (< N)."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k=9,
                                    oxide_threshold=THRESHOLD, seed=0, min_k=3)
        # full_groups=9, active_groups=7 → expected ~N*7/9
        expected_approx = N * 7 // 9
        assert len(df) < N, \
            f"max_k=9 should produce fewer than {N} rows with full_groups normalization, got {len(df)}"
        assert len(df) >= expected_approx - 10, \
            f"max_k=9 produced too few rows: {len(df)} < {expected_approx - 10}"

    def test_sparse_max9_survival_rate_below_full(self):
        """full_groups normalization: max_k=9 produces 7/9 of N, max_k=11 produces ~N."""
        raw_full = sample_simplex(OXIDES_11, N, seed=0)
        full_after = _apply_threshold(raw_full, THRESHOLD)
        # full simplex still loses a tiny number of rows
        assert len(full_after) >= N * 0.99, \
            f"full simplex unexpectedly lost many rows: {len(full_after)}"

        sparse = _sample_sparse_subsets(OXIDES_11, N, max_k=9,
                                        oxide_threshold=THRESHOLD, seed=0, min_k=3)
        assert len(sparse) < N, \
            f"sparse max_k=9 should return fewer than {N} rows with full_groups normalization, got {len(sparse)}"

    def test_sparse_max9_no_row_exceeds_max_k(self):
        """No row returned by sparse sampler with max_k=9 can have n_oxides > 9."""
        df = _sample_sparse_subsets(OXIDES_11, N, max_k=9,
                                    oxide_threshold=THRESHOLD, seed=42, min_k=3)
        n_ox = (df > 0).sum(axis=1)
        bad = (n_ox > 9).sum()
        assert bad == 0, f"{bad} rows exceed max_k=9"

    def test_sparse_max11_matches_full_n_oxides_distribution(self):
        """Sparse sampler with max_k=11 should include n=11 rows (unlike max_k=9)."""
        sparse11 = _sample_sparse_subsets(OXIDES_11, N, max_k=11,
                                          oxide_threshold=THRESHOLD, seed=0, min_k=3)
        sparse9 = _sample_sparse_subsets(OXIDES_11, N, max_k=9,
                                         oxide_threshold=THRESHOLD, seed=0, min_k=3)
        n_ox_11 = (sparse11 > 0).sum(axis=1)
        n_ox_9 = (sparse9 > 0).sum(axis=1)
        assert (n_ox_11 == 11).any(), "sparse max_k=11 should produce some n=11 rows"
        assert not (n_ox_9 == 11).any(), "sparse max_k=9 must not produce n=11 rows"


# ── Key behavior: sparse does NOT post-filter full simplex ───────────────────

class TestSparseIsNotPostFilter:
    """Proves sparse sampling is NOT '11-oxide simplex then filter ≤9'.

    If it were post-filtering:
      - survival rate would be ~full rate (since ~92% of full-simplex has n>=9)
      - the distribution of kept rows would look like full simplex filtered

    Sparse sampling instead generates subsets directly, so:
      - low-n rows (n=3..5) appear with reasonable frequency
      - survival rate is noticeably lower
    """

    def test_low_n_oxides_present_in_sparse_but_not_full(self):
        """Full simplex produces almost no n=3 rows; sparse max_k=9 produces many."""
        raw_full = sample_simplex(OXIDES_11, N, seed=0)
        full_after = _apply_threshold(raw_full, THRESHOLD)
        n3_full = (( full_after > 0).sum(axis=1) <= 5).sum()

        sparse = _sample_sparse_subsets(OXIDES_11, N, max_k=9,
                                        oxide_threshold=THRESHOLD, seed=0, min_k=3)
        n3_sparse = ((sparse > 0).sum(axis=1) <= 5).sum()

        # sparse should have at least 50x more low-n rows than full simplex
        assert n3_sparse > n3_full * 50, (
            f"sparse low-n count {n3_sparse} not much larger than full {n3_full}; "
            "sparse sampler may be behaving like a post-filter"
        )

    def test_sparse_low_n_fraction_at_least_10pct(self):
        """At least 10% of sparse max_k=9 rows should have n_oxides ≤ 5."""
        sparse = _sample_sparse_subsets(OXIDES_11, N, max_k=9,
                                        oxide_threshold=THRESHOLD, seed=0, min_k=3)
        n_ox = (sparse > 0).sum(axis=1)
        frac = (n_ox <= 5).sum() / len(sparse)
        assert frac >= 0.10, \
            f"only {frac:.2%} of sparse rows have n<=5; expected >=10%"


# ── Quantitative survival rates (informational bounds) ───────────────────────

class TestSurvivalRateBounds:
    """Bounds on how many rows survive for each max_k."""

    @pytest.mark.parametrize("max_k,expected_min_rate,expected_max_rate", [
        # Threshold=1% alone drops very few rows for any max_k,
        # because even k=3 Dirichlet gives ~33% per component >> 1%.
        # The main source of candidate reduction in the app is the eps_r filter,
        # not the threshold.  These bounds reflect the observed ~90-100% survival.
        # With full_groups=9 normalization, total = active_groups/9 * N.
        # Rates: k=3→1/9≈0.11, k=5→3/9≈0.33, k=7→5/9≈0.56, k=9→7/9≈0.78, k=11→9/9=1.00
        # Allow ±0.05 tolerance for remainder distribution.
        (3,  0.10, 0.15),
        (5,  0.30, 0.40),
        (7,  0.53, 0.63),
        (9,  0.74, 0.84),
        (11, 0.95, 1.00),
    ])
    def test_survival_within_expected_range(
        self, max_k, expected_min_rate, expected_max_rate
    ):
        df = _sample_sparse_subsets(OXIDES_11, N, max_k=max_k,
                                    oxide_threshold=THRESHOLD, seed=0, min_k=3)
        rate = len(df) / N
        assert expected_min_rate <= rate <= expected_max_rate, (
            f"max_k={max_k}: survival rate {rate:.3f} outside "
            f"[{expected_min_rate}, {expected_max_rate}]"
        )
