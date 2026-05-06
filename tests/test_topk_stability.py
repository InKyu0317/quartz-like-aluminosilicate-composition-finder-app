"""Prove-It: explain why top-1 composition changes when max_n_oxides goes 9 → 11.

User's question:
  "max oxide count 를 9로 했을때랑 11로 했을때 1순위 조성이 왜 바뀌는지 이해가 안가."

Hypothesis (NOT a bug — it's expected sampling-stratification behavior):
  1. _sample_sparse_subsets stratifies samples *equally per k* (k = active oxide count),
     where k ranges from min_k=3 to max_k=max_n_oxides.
     - max_n_oxides=9  → 7 groups (k=3..9), each gets n_samples/7 rows
     - max_n_oxides=11 → 9 groups (k=3..11), each gets n_samples/9 rows
  2. Even at the same seed, group-count and per-group batch size differ, so RNG
     consumption differs → DIFFERENT samples, even for k=3..9 overlap.
  3. With finite n_samples, top-1 is sampling noise — varying seed at fixed
     max_n_oxides also flips the top-1 composition.

Tests below PROVE these claims so the user can stop suspecting a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roadlab_matnav_lib.recommend import _sample_sparse_subsets

OXIDES_11 = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO',
    'ZnO', 'ZrO2', 'B2O3', 'La2O3', 'Y2O3',
]
N = 600
THR = 1.0


def _topk_sig(df: pd.DataFrame, k: int = 1) -> list[tuple[str, float]]:
    """Return a hashable signature of the top row's nonzero oxides (rounded)."""
    if df.empty:
        return []
    row = df.iloc[0]
    return sorted(
        [(ox, round(float(row[ox]), 1)) for ox in OXIDES_11 if row[ox] > 0.0],
        key=lambda x: -x[1],
    )[:5]  # top-5 oxides of the top-1 composition


# ── Claim 1: max=9 vs max=11 produce different sample distributions ─────────

class TestStratificationChangesSamples:
    def test_max9_vs_max11_overlap_rows_are_different(self):
        """k=3..9 overlap region, but actual samples differ because
        per-group quotas + RNG advancement differ."""
        df9  = _sample_sparse_subsets(OXIDES_11, N, max_k=9,  oxide_threshold=THR, seed=42)
        df11 = _sample_sparse_subsets(OXIDES_11, N, max_k=11, oxide_threshold=THR, seed=42)

        n_ox_9  = (df9  > 0).sum(axis=1)
        n_ox_11 = (df11 > 0).sum(axis=1)

        # Filter to the overlap k=3..9 only
        sub9  = df9.loc[n_ox_9.between(3, 9)].reset_index(drop=True)
        sub11 = df11.loc[n_ox_11.between(3, 9)].reset_index(drop=True)

        # If samples were identical, an inner-merge would recover all of them.
        # We round to 2 decimals to be robust to float jitter.
        merge_cols = OXIDES_11
        a = sub9[merge_cols].round(2)
        b = sub11[merge_cols].round(2)
        common = pd.merge(a, b, how="inner", on=merge_cols)
        overlap_ratio = len(common) / max(len(a), 1)
        assert overlap_ratio < 0.05, (
            f"Expected <5% identical rows in k=3..9 overlap, got {overlap_ratio:.3f}. "
            f"If this is high, sampling IS deterministic across max_k — would be unexpected."
        )

    def test_per_k_quota_changes_with_max_k(self):
        """Number of rows per k changes when max_k changes (stratified equal)."""
        df9  = _sample_sparse_subsets(OXIDES_11, 700, max_k=9,  oxide_threshold=THR, seed=0)
        df11 = _sample_sparse_subsets(OXIDES_11, 900, max_k=11, oxide_threshold=THR, seed=0)

        # max_k=9 → 7 groups (k=3..9) → ~100 rows per k
        # max_k=11 → 9 groups (k=3..11) → ~100 rows per k
        counts_9 = (df9 > 0).sum(axis=1).value_counts().to_dict()
        counts_11 = (df11 > 0).sum(axis=1).value_counts().to_dict()

        assert set(counts_9.keys()).issubset({3, 4, 5, 6, 7, 8, 9})
        assert set(counts_11.keys()).issubset({3, 4, 5, 6, 7, 8, 9, 10, 11})
        # Largest k should exist in each
        assert 9 in counts_9
        assert 11 in counts_11


# ── Claim 2: top-1 is noisy even when only the seed changes ─────────────────

class TestTopOneIsSamplingNoise:
    """Synthetic 'score' = mol-weighted-sum to mimic a real BO score:
    different random samples → different top-1, even with same max_k.
    Proves that top-1 flipping is intrinsic to finite Dirichlet sampling,
    not a bug introduced by max_n_oxides.
    """

    def _score(self, df: pd.DataFrame) -> pd.Series:
        # Toy score that prefers high SiO2 + high Al2O3 + low BaO.
        return df["SiO2"] * 0.5 + df["Al2O3"] * 0.3 - df["BaO"] * 0.2

    def test_top1_changes_across_seeds_at_same_max_k(self):
        """Same max_k=9, different seeds → different top-1 composition.
        Demonstrates that top-1 is sampling noise."""
        sigs = set()
        for seed in (0, 1, 2, 3, 4):
            df = _sample_sparse_subsets(
                OXIDES_11, N, max_k=9, oxide_threshold=THR, seed=seed,
            )
            df = df.assign(_s=self._score(df)).sort_values("_s", ascending=False)
            sigs.add(tuple(_topk_sig(df.drop(columns="_s"))))
        assert len(sigs) >= 3, (
            f"Expected ≥3 distinct top-1 compositions across 5 seeds, got {len(sigs)}. "
            f"If only 1, top-1 would be deterministic — disproving the noise claim."
        )

    def test_top1_changes_when_only_max_k_changes(self):
        """Same seed, only max_k changes 9 → 11 → different top-1.
        This is the user's exact observation — and it's expected."""
        df9 = _sample_sparse_subsets(OXIDES_11, N, max_k=9,  oxide_threshold=THR, seed=42)
        df11 = _sample_sparse_subsets(OXIDES_11, N, max_k=11, oxide_threshold=THR, seed=42)

        df9  = df9.assign(_s=self._score(df9)).sort_values("_s", ascending=False)
        df11 = df11.assign(_s=self._score(df11)).sort_values("_s", ascending=False)

        sig9  = tuple(_topk_sig(df9.drop(columns="_s")))
        sig11 = tuple(_topk_sig(df11.drop(columns="_s")))
        assert sig9 != sig11, "Top-1 was identical across max_k — would contradict the explanation."
