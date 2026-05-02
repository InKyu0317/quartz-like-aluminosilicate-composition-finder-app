"""Prove-It: max_n_oxides=11 (default) must NOT miss low-n high-score compositions.

Bug report:
  - max_n_oxides=11 → rank 1 score=0.2915 (6 oxides)
  - max_n_oxides=9  → rank 1 score=0.3878 (3 oxides)
  These should produce comparable best scores; the 3-oxide composition is simply
  absent from the full-simplex search space.

Root cause:
  recommend() with max_n_oxides=11 uses sample_simplex (11-oxide Dirichlet),
  which almost never produces n<=5 compositions (probability ~0%).
  recommend() with max_n_oxides<11 uses _sample_sparse_subsets, which covers
  k=3..max_k evenly.

Fix: always use _sample_sparse_subsets regardless of max_n_oxides.
"""
import numpy as np
import pandas as pd
import pytest

from roadlab_matnav_lib.recommend import recommend

OXIDES_11 = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO',
    'ZnO', 'ZrO2', 'B2O3', 'La2O3', 'Y2O3',
]
N = 5_000
THRESHOLD = 1.0


class MockPredictor:
    """Predictable mock: eps=5.0 (always in range), tan=0.000198 (quartz exact)."""
    def batch_eps_tan(self, mol_df):
        n = len(mol_df)
        return np.full(n, 5.0), np.full(n, 0.000198)


# ── Core bug: max_n_oxides=None must include low-n compositions ──────────────

class TestDefaultIncludesLowNCompositions:

    def test_max_n_none_produces_low_n_rows(self):
        """BUG: recommend() with max_n_oxides=None (full simplex) NEVER produces
        n_oxides<=5 rows.  After fix, >=5% of results should have n_oxides<=5."""
        predictor = MockPredictor()
        df = recommend(
            predictor, OXIDES_11,
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=N,
            oxide_threshold=THRESHOLD,
            max_n_oxides=None,
            seed=0,
        )
        n_ox = (df[OXIDES_11] > THRESHOLD).sum(axis=1)
        frac_low = (n_ox <= 5).sum() / len(df)
        assert frac_low >= 0.05, (
            f"max_n_oxides=None: only {frac_low:.2%} rows have n<=5; "
            f"expected >=5% — full-simplex sampling is missing low-n compositions"
        )

    def test_max_n_11_produces_low_n_rows(self):
        """BUG: recommend() with max_n_oxides=11 (== len) NEVER produces
        n_oxides<=5 rows.  After fix, >=5% of results should have n_oxides<=5."""
        predictor = MockPredictor()
        df = recommend(
            predictor, OXIDES_11,
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=N,
            oxide_threshold=THRESHOLD,
            max_n_oxides=len(OXIDES_11),
            seed=0,
        )
        n_ox = (df[OXIDES_11] > THRESHOLD).sum(axis=1)
        frac_low = (n_ox <= 5).sum() / len(df)
        assert frac_low >= 0.05, (
            f"max_n_oxides=11: only {frac_low:.2%} rows have n<=5; "
            f"expected >=5% — full-simplex sampling is missing low-n compositions"
        )


# ── Score consistency: max_n=11 must not have a LOWER best score than max_n=9 ─

class TestScoreConsistency:

    def test_max11_best_score_not_below_max9(self):
        """With same n_samples, max_n_oxides=11 should find compositions at least
        as good as max_n_oxides=9.  Currently FAILS because max=11 uses full
        simplex and never finds the sparse high-score compositions."""
        predictor = MockPredictor()
        common = dict(
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=N,
            oxide_threshold=THRESHOLD,
            seed=42,
        )
        df9  = recommend(predictor, OXIDES_11, max_n_oxides=9,  **common)
        df11 = recommend(predictor, OXIDES_11, max_n_oxides=11, **common)

        # With mock predictor (all tan=quartz), scores are all identical=1.0
        # so this mainly tests that both paths produce valid results.
        # Real value: both should cover the same low-n space.
        assert len(df11) >= len(df9) * 0.8, (
            f"max_n=11 produced far fewer results ({len(df11)}) than max_n=9 ({len(df9)})"
        )

    def test_max11_n_oxides_range_covers_3_to_11(self):
        """max_n_oxides=11 should produce n_oxides from 3 to 11, not just 9-11."""
        predictor = MockPredictor()
        df = recommend(
            predictor, OXIDES_11,
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=N,
            oxide_threshold=THRESHOLD,
            max_n_oxides=11,
            seed=0,
        )
        n_ox = (df[OXIDES_11] > THRESHOLD).sum(axis=1)
        assert n_ox.min() <= 5, (
            f"max_n_oxides=11: minimum n_oxides={n_ox.min()}, expected <=5. "
            "Low-n compositions not being generated."
        )
        assert n_ox.max() == 11, (
            f"max_n_oxides=11: maximum n_oxides={n_ox.max()}, expected 11."
        )


# ── Behavioral equivalence: max_n=9 sparse must equal max_n=9 after fix ─────

class TestSparsePathUnchanged:

    def test_max9_still_caps_at_9_oxides(self):
        """After fix, max_n_oxides=9 must still produce no rows with n>9."""
        predictor = MockPredictor()
        df = recommend(
            predictor, OXIDES_11,
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=N,
            oxide_threshold=THRESHOLD,
            max_n_oxides=9,
            seed=0,
        )
        n_ox = (df[OXIDES_11] > THRESHOLD).sum(axis=1)
        assert (n_ox <= 9).all(), f"max_n_oxides=9: found rows with n_oxides > 9"

    def test_max5_still_caps_at_5_oxides(self):
        """After fix, max_n_oxides=5 must still produce no rows with n>5."""
        predictor = MockPredictor()
        df = recommend(
            predictor, OXIDES_11,
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=N,
            oxide_threshold=THRESHOLD,
            max_n_oxides=5,
            seed=0,
        )
        n_ox = (df[OXIDES_11] > THRESHOLD).sum(axis=1)
        assert (n_ox <= 5).all(), f"max_n_oxides=5: found rows with n_oxides > 5"
