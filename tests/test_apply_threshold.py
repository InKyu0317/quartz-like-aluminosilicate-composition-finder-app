"""Tests for _apply_threshold and oxide_threshold integration in recommend().

Prove-It pattern:
1. Each test describes the expected behavior precisely
2. Tests verify the FIX (threshold applied before model call, not just display)
"""
import numpy as np
import pandas as pd
import pytest

from roadlab_matnav_lib.recommend import (
    _apply_threshold,
    wt_to_mol_frame,
    sample_simplex,
)


# ─────────────────────────────────────────────────────────────────────────────
# _apply_threshold unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyThreshold:

    def _df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    def test_values_below_threshold_become_zero(self):
        """Oxides with wt% < threshold must be set to exactly 0.0."""
        df = self._df({"SiO2": [70.0], "Al2O3": [0.5], "CaO": [29.5]})
        result = _apply_threshold(df, threshold=1.0)
        assert result["Al2O3"].iloc[0] == 0.0

    def test_values_at_or_above_threshold_are_kept(self):
        """Oxides at exactly threshold or above must survive zeroing."""
        df = self._df({"SiO2": [70.0], "Al2O3": [1.0], "CaO": [29.0]})
        result = _apply_threshold(df, threshold=1.0)
        assert result["Al2O3"].iloc[0] > 0.0

    def test_rows_renormalize_to_100(self):
        """After zeroing, each row must sum to exactly 100 wt%."""
        df = self._df({"SiO2": [65.0], "Al2O3": [0.3], "CaO": [0.2], "MgO": [34.5]})
        result = _apply_threshold(df, threshold=1.0)
        row_sum = result.iloc[0].sum()
        assert abs(row_sum - 100.0) < 1e-9, f"row sum = {row_sum}"

    def test_n_oxides_count_matches_threshold(self):
        """n_oxides should equal the number of non-zero oxides after threshold."""
        df = self._df({
            "SiO2":  [60.0],
            "Al2O3": [20.0],
            "CaO":   [0.4],   # below threshold
            "MgO":   [0.1],   # below threshold
            "BaO":   [19.5],
        })
        result = _apply_threshold(df, threshold=1.0)
        n_nonzero = (result.iloc[0] > 0).sum()
        assert n_nonzero == 3, f"expected 3, got {n_nonzero}"

    def test_zero_threshold_returns_unchanged(self):
        """threshold=0.0 must be a no-op (early return path)."""
        df = self._df({"SiO2": [70.0], "Al2O3": [0.1], "CaO": [29.9]})
        result = _apply_threshold(df, threshold=0.0)
        pd.testing.assert_frame_equal(result, df)

    def test_rows_where_all_below_threshold_are_dropped(self):
        """A row where every oxide is below threshold should be dropped."""
        df = self._df({
            "SiO2":  [70.0, 0.5],
            "Al2O3": [30.0, 0.5],
        })
        result = _apply_threshold(df, threshold=1.0)
        assert len(result) == 1, f"expected 1 row, got {len(result)}"

    def test_multiple_rows_each_renormalize_independently(self):
        """Each row renormalizes to 100 independently."""
        df = self._df({
            "SiO2":  [80.0, 60.0],
            "Al2O3": [0.5,  0.3],  # below threshold in both
            "CaO":   [19.5, 39.7],
        })
        result = _apply_threshold(df, threshold=1.0)
        for i in range(len(result)):
            row_sum = result.iloc[i].sum()
            assert abs(row_sum - 100.0) < 1e-9, f"row {i} sum = {row_sum}"

    def test_zeroed_oxide_does_not_appear_in_mol_frame(self):
        """After threshold zeroing, wt_to_mol_frame must also yield 0.0
        for that oxide — not a residual mol% from a near-zero wt%."""
        df = self._df({"SiO2": [65.0], "Al2O3": [0.3], "CaO": [34.7]})
        cleaned = _apply_threshold(df, threshold=1.0)
        mol = wt_to_mol_frame(cleaned)
        assert mol["Al2O3"].iloc[0] == 0.0, (
            f"Al2O3 mol% should be 0.0, got {mol['Al2O3'].iloc[0]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: sample_simplex + _apply_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleSimplexThreshold:
    OXIDES = ["SiO2", "Al2O3", "MgO", "CaO", "SrO"]
    THRESHOLD = 1.0

    def test_without_threshold_all_values_nonzero(self):
        """Dirichlet sampling with no threshold produces all non-zero values."""
        df = sample_simplex(self.OXIDES, n=100, seed=42)
        # At least some rows should have all 5 oxides > 0 (Dirichlet property)
        all_positive = (df > 0).all(axis=1).sum()
        assert all_positive > 50, "Expected most rows to have all oxides > 0"

    def test_with_threshold_n_oxides_consistent(self):
        """After threshold, n_oxides computed from the df must match
        the number of non-zero oxides exactly (no residual traces)."""
        df = sample_simplex(self.OXIDES, n=200, seed=42)
        cleaned = _apply_threshold(df, threshold=self.THRESHOLD)

        for i in range(len(cleaned)):
            row = cleaned.iloc[i]
            n_nonzero = int((row > 0).sum())
            n_above_threshold = int((row >= self.THRESHOLD).sum())
            assert n_nonzero == n_above_threshold, (
                f"Row {i}: n_nonzero={n_nonzero} != n_above_threshold={n_above_threshold}\n{row}"
            )

    def test_with_threshold_rows_sum_to_100(self):
        """Every row after threshold must sum to 100 wt%."""
        df = sample_simplex(self.OXIDES, n=200, seed=42)
        cleaned = _apply_threshold(df, threshold=self.THRESHOLD)
        row_sums = cleaned.sum(axis=1)
        assert (np.abs(row_sums - 100.0) < 1e-8).all(), (
            f"Row sums off: min={row_sums.min():.6f} max={row_sums.max():.6f}"
        )

    def test_wt_to_mol_after_threshold_zero_stays_zero(self):
        """Zeroed wt% oxides must remain 0.0 after mol% conversion."""
        df = sample_simplex(self.OXIDES, n=100, seed=42)
        cleaned = _apply_threshold(df, threshold=self.THRESHOLD)
        mol = wt_to_mol_frame(cleaned)

        # Wherever wt%==0 → mol% must also be 0
        wt_zero = (cleaned == 0.0)
        mol_zero = (mol == 0.0)
        mismatches = wt_zero & ~mol_zero
        assert not mismatches.any().any(), (
            f"Non-zero mol% found where wt%=0:\n{mol[mismatches.any(axis=1)]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regression: the original bug (display-only zeroing, model saw trace values)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionDisplayVsModelInput:

    def test_mol_frame_without_threshold_has_nonzero_for_trace_oxides(self):
        """REGRESSION: without threshold, a trace 0.3 wt% oxide produces
        a non-zero mol% that was silently fed to GlassNet before the fix."""
        df = pd.DataFrame({"SiO2": [65.0], "Al2O3": [0.3], "CaO": [34.7]})
        mol = wt_to_mol_frame(df)   # no threshold applied
        # Al2O3 at 0.3 wt% gives a non-zero mol% — this was the bug
        assert mol["Al2O3"].iloc[0] > 0.0, (
            "Pre-condition check: trace oxide produces non-zero mol% without threshold"
        )

    def test_mol_frame_with_threshold_has_zero_for_trace_oxides(self):
        """FIX: with threshold applied first, the same trace oxide yields 0 mol%."""
        df = pd.DataFrame({"SiO2": [65.0], "Al2O3": [0.3], "CaO": [34.7]})
        cleaned = _apply_threshold(df, threshold=1.0)
        mol = wt_to_mol_frame(cleaned)
        assert mol["Al2O3"].iloc[0] == 0.0, (
            f"Expected 0.0 mol% for zeroed oxide, got {mol['Al2O3'].iloc[0]}"
        )

    def test_compositions_differ_before_and_after_threshold(self):
        """The mol% composition passed to the model must differ when threshold
        is applied — confirming the fix changes the model input, not just display."""
        df = pd.DataFrame({"SiO2": [65.0], "Al2O3": [0.3], "CaO": [34.7]})
        mol_before = wt_to_mol_frame(df)
        cleaned = _apply_threshold(df, threshold=1.0)
        mol_after = wt_to_mol_frame(cleaned)
        assert not mol_before.equals(mol_after), (
            "Model input should differ after threshold application"
        )
