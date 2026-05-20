"""Prove-It (C2 fix): recommend() internal score must rank lower tan_delta higher.

Bug:  tan_component = raw tan_delta  →  higher tan → higher score → wrong rank.
Fix:  tan_component = 1/(1+tan)      →  lower  tan → higher score → correct rank.

These tests FAIL before the fix and PASS after.
"""
from __future__ import annotations

import numpy as np
import pytest

from roadlab_matnav_lib.recommend import recommend

OXIDES_5 = ["SiO2", "Al2O3", "MgO", "CaO", "B2O3"]


class _MockPredictorSplit:
    """Assigns tan_delta based on SiO2 mol%: >50 → low tan (good), ≤50 → high tan."""

    def batch_eps_tan(self, mol_df):
        n = len(mol_df)
        eps = np.full(n, 5.0)
        sio2 = mol_df["SiO2"].to_numpy() if "SiO2" in mol_df.columns else np.zeros(n)
        tan = np.where(sio2 > 50.0, 1e-4, 1e-3)
        return eps, tan


class TestScoreDirectionLowerTanBetter:

    def _call(self, n: int = 400, seed: int = 42) -> "pd.DataFrame":
        import pandas as pd  # noqa: F401 — type hint only
        return recommend(
            _MockPredictorSplit(),
            OXIDES_5,
            eps_r_range=(3.0, 15.0),
            tan_delta_range=(0.0, 1.0),
            n_samples=n,
            oxide_threshold=1.0,
            seed=seed,
        )

    def test_score_column_present(self):
        """recommend() output must contain a 'score' column."""
        df = self._call()
        assert "score" in df.columns, "recommend() did not return a 'score' column"

    def test_lower_tan_gets_higher_score(self):
        """Compositions with lower tan_delta must receive a strictly higher score.

        With the BROKEN formula (tan_component = raw tan_delta):
          - tan=1e-3 → score=1e-3  (high, ranks first)  ← wrong
          - tan=1e-4 → score=1e-4  (low, ranks last)   ← wrong

        With the FIXED formula (tan_component = 1/(1+tan)):
          - tan=1e-4 → 1/1.0001 ≈ 0.9999  (high score) ← correct
          - tan=1e-3 → 1/1.001  ≈ 0.999   (low  score) ← correct
        """
        df = self._call()
        assert len(df) > 20, f"too few results: {len(df)}"

        low_tan  = df[df["tan_delta"] < 2e-4]
        high_tan = df[df["tan_delta"] > 5e-4]
        assert len(low_tan) > 0 and len(high_tan) > 0, (
            "mock predictor must produce both low- and high-tan groups; "
            f"got {len(low_tan)} low, {len(high_tan)} high"
        )

        mean_score_low  = float(low_tan["score"].mean())
        mean_score_high = float(high_tan["score"].mean())
        assert mean_score_low > mean_score_high, (
            f"FAIL (C2 bug): low-tan mean score {mean_score_low:.6f} "
            f"<= high-tan mean score {mean_score_high:.6f}. "
            "Lower tan_delta should produce a HIGHER score."
        )

    def test_returned_df_sorted_low_tan_first(self):
        """After recommend()'s internal sort, early rows should have lower tan_delta
        than late rows (because lower tan → higher score → sorted to front)."""
        df = self._call(n=300)
        assert len(df) >= 20

        top_q    = df.head(len(df) // 4)["tan_delta"].mean()
        bottom_q = df.tail(len(df) // 4)["tan_delta"].mean()
        assert top_q < bottom_q, (
            f"Top-quarter mean tan_delta ({top_q:.6f}) >= bottom-quarter ({bottom_q:.6f}). "
            "recommend() should return lower-tan rows first."
        )

    def test_score_monotone_with_tan(self):
        """score must be a strictly decreasing function of tan_delta (w_tan=1, w_eps=0)."""
        df = self._call()
        # Pick two distinct tan levels
        low_row  = df[df["tan_delta"] < 2e-4].head(1)
        high_row = df[df["tan_delta"] > 5e-4].head(1)
        if low_row.empty or high_row.empty:
            pytest.skip("not enough variety for this check")

        s_low  = float(low_row["score"].iloc[0])
        s_high = float(high_row["score"].iloc[0])
        assert s_low > s_high, (
            f"score({low_row['tan_delta'].iloc[0]:.2e}) = {s_low:.6f} "
            f"<= score({high_row['tan_delta'].iloc[0]:.2e}) = {s_high:.6f}"
        )
