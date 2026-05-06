"""Composition recommender — sampling and filtering on an oxide simplex.

Public API:

- :func:`sample_simplex` — uniform Dirichlet samples on a chosen subset of
  oxides, optionally constrained by per-oxide bounding boxes (rejection
  sampling).
- :func:`recommend` — sample → batch-predict (ε_r, tan δ, ...) → filter →
  score → sort. (T19.2 — implemented in a follow-up step.)
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import oxides

__all__ = ["sample_simplex", "recommend", "wt_to_mol_frame"]

BoundsLike = Mapping[str, "tuple[float, float]"]


def wt_to_mol_frame(df_wt: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of wt% compositions to mol% (vectorized).

    Each column is divided by its oxide molecular weight, then each row is
    normalised to sum to 100. Vectorized over all rows at once — avoids the
    O(n) Python loop of the previous ``iterrows()`` implementation.
    """
    mw_map = {ox: oxides.info(ox).mw for ox in df_wt.columns}
    mw_arr = np.array([mw_map[c] for c in df_wt.columns], dtype=float)
    mol = df_wt.to_numpy(dtype=float) / mw_arr  # broadcast divide by MW
    row_sums = mol.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
    mol = mol / row_sums * 100.0
    return pd.DataFrame(mol, columns=df_wt.columns)


def sample_simplex(
    active_oxides: Sequence[str],
    n: int,
    *,
    fixed: Optional[BoundsLike] = None,
    seed: Optional[int] = None,
    max_attempts_factor: int = 50,
) -> pd.DataFrame:
    """Draw ``n`` compositions uniformly on the simplex spanned by
    ``active_oxides`` (Dirichlet α=1, scaled to sum to 100).

    Parameters
    ----------
    active_oxides:
        Ordered list of oxide formulas (e.g. ``['SiO2', 'Al2O3', 'Na2O']``).
        All entries must be in :func:`roadlab_matnav_lib.oxides.list_supported`.
    n:
        Number of accepted samples to return.
    fixed:
        Optional mapping ``{oxide: (lo, hi)}`` of inclusive bounding-box
        constraints in wt%.  Implemented via rejection sampling.
    seed:
        RNG seed for reproducibility.
    max_attempts_factor:
        Hard cap on rejection sampling — total attempts ≤ ``n * factor``.
        Raises :class:`ValueError` if not enough samples are accepted.

    Returns
    -------
    pandas.DataFrame
        Shape ``(n, len(active_oxides))``, columns ordered as
        ``active_oxides``. Each row sums to ~100.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(active_oxides) < 2:
        raise ValueError(
            f"need at least 2 oxides, got {len(active_oxides)}: {active_oxides!r}"
        )
    if len(set(active_oxides)) != len(active_oxides):
        raise ValueError(f"duplicate oxide in active_oxides: {active_oxides!r}")

    supported = set(oxides.list_supported())
    for ox in active_oxides:
        if ox not in supported:
            raise KeyError(f"unsupported oxide: {ox!r}")

    fixed = dict(fixed or {})
    for ox in fixed:
        if ox not in active_oxides:
            raise KeyError(
                f"fixed oxide {ox!r} not in active_oxides {list(active_oxides)!r}"
            )

    rng = np.random.default_rng(seed)
    K = len(active_oxides)
    accepted: list[np.ndarray] = []
    max_total = int(n * max_attempts_factor)
    drawn = 0
    batch = max(n, 256)

    while len(accepted) < n and drawn < max_total:
        take = min(batch, max_total - drawn)
        samples = rng.dirichlet(np.ones(K), size=take) * 100.0
        drawn += take
        if fixed:
            ok = np.ones(take, dtype=bool)
            for ox, (lo, hi) in fixed.items():
                idx = list(active_oxides).index(ox)
                col = samples[:, idx]
                ok &= (col >= float(lo)) & (col <= float(hi))
            samples = samples[ok]
        for row in samples:
            accepted.append(row)
            if len(accepted) >= n:
                break

    if len(accepted) < n:
        raise ValueError(
            f"could only accept {len(accepted)}/{n} samples within "
            f"{max_total} attempts (constraints too tight?)"
        )

    arr = np.asarray(accepted[:n], dtype=float)
    return pd.DataFrame(arr, columns=list(active_oxides))


def _apply_threshold(df_wt: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Zero out oxide values below ``threshold`` wt% and renormalize each row
    to 100 wt%.

    Dirichlet sampling always assigns small non-zero values to every oxide.
    Applying a threshold before model calls ensures predictions are made on
    compositions where minor-trace oxides are truly absent, not just small.
    Rows where all values fall below the threshold are dropped.
    """
    if threshold <= 0.0:
        return df_wt
    cleaned = df_wt.where(df_wt >= threshold, 0.0)
    row_sums = cleaned.sum(axis=1)
    valid = row_sums > 0
    cleaned = cleaned.loc[valid]
    row_sums = row_sums.loc[valid]
    return cleaned.div(row_sums, axis=0).mul(100.0).reset_index(drop=True)


def _sample_sparse_subsets(
    active_oxides: Sequence[str],
    n_samples: int,
    max_k: int,
    oxide_threshold: float,
    seed: Optional[int],
    min_k: int = 3,
) -> pd.DataFrame:
    """Sample ``n_samples`` compositions with exactly equal counts per oxide number.

    Stratified by k (= intended subset size, min_k..max_k):
    - Per-k quota is fixed to ``n_samples // (K - min_k + 1)`` where K is the
      total oxide pool size — independent of ``max_k``.  This means the sampling
      density for k=3..min(max_k1, max_k2) is identical regardless of which
      max_k is chosen, so top-1 candidates in the overlapping k range are stable.
    - Total output ≤ n_samples when max_k < K (recommend() batching compensates).
    - Oversamples each k until quota is filled, so threshold dropout does not
      create imbalance between groups.
    - SiO2 is always included when present.
    - Rows where threshold zeroed SiO2 or left fewer than min_k oxides are
      discarded and replaced.
    """
    min_k = max(min_k, 2)
    rng = np.random.default_rng(seed)
    oxides_list = list(active_oxides)
    K = len(oxides_list)
    sio2_idx = oxides_list.index("SiO2") if "SiO2" in oxides_list else None
    other_idx = np.array([i for i in range(K) if i != sio2_idx])

    # Fix quota per k based on the FULL oxide pool (K - min_k + 1 groups),
    # not the active max_k.  This makes per-k density independent of max_k:
    # changing max_oxide_count no longer redistributes samples between k=3..9,
    # so top-1 candidates in the overlapping k range stay consistent.
    # Trade-off: total output = (max_k - min_k + 1) * quota_base <= n_samples
    # when max_k < K.  recommend() iterative batching handles any shortfall.
    full_groups = K - min_k + 1
    quota_base = n_samples // full_groups
    remainder = n_samples % full_groups

    def _generate_batch_for_k(k: int, batch_size: int) -> pd.DataFrame:
        """Vectorized: generate batch_size rows each with k-oxide Dirichlet."""
        n_others = min(k - 1, len(other_idx)) if sio2_idx is not None else k
        actual_k = (n_others + 1) if sio2_idx is not None else k

        if sio2_idx is not None:
            if n_others > 0:
                # Random permutation of other_idx for each row, take first n_others
                perms = rng.permuted(
                    np.tile(other_idx, (batch_size, 1)), axis=1
                )
                chosen = perms[:, :n_others]  # (batch_size, n_others)
                subset_idx = np.column_stack(
                    [np.full(batch_size, sio2_idx), chosen]
                )  # (batch_size, actual_k)
            else:
                subset_idx = np.full((batch_size, 1), sio2_idx)
        else:
            perms = rng.permuted(np.tile(np.arange(K), (batch_size, 1)), axis=1)
            subset_idx = perms[:, :k]

        dirichlet = rng.dirichlet(np.ones(actual_k), size=batch_size) * 100.0
        rows = np.zeros((batch_size, K), dtype=float)
        np.put_along_axis(rows, subset_idx, dirichlet, axis=1)

        df = pd.DataFrame(rows, columns=oxides_list)
        if oxide_threshold > 0.0:
            df = _apply_threshold(df, oxide_threshold)
        # Keep only rows with min_k active oxides and SiO2 intact
        valid = (df > 0).sum(axis=1) >= min_k
        if sio2_idx is not None:
            valid &= df["SiO2"] > 0
        return df.loc[valid].reset_index(drop=True)

    all_dfs: list[pd.DataFrame] = []
    for gi, k in enumerate(range(min_k, max_k + 1)):
        target = quota_base + (1 if gi < remainder else 0)
        collected: list[pd.DataFrame] = []
        total = 0
        oversample = 3  # initial oversampling factor
        while total < target:
            needed = target - total
            batch_size = max(needed * oversample, 256)
            chunk = _generate_batch_for_k(k, batch_size)
            collected.append(chunk)
            total += len(chunk)
            oversample = max(oversample, int(batch_size / max(len(chunk), 1)) + 1)
        group_df = pd.concat(collected, ignore_index=True).head(target)
        all_dfs.append(group_df)

    return pd.concat(all_dfs, ignore_index=True)


def recommend(
    predictor,
    active_oxides: Sequence[str],
    *,
    eps_r_range: "tuple[float, float] | None" = None,
    tan_delta_range: "tuple[float, float] | None" = None,
    n_samples: int = 1000,
    fixed: Optional[BoundsLike] = None,
    seed: Optional[int] = None,
    extra_props: Iterable[str] = (),
    max_attempts_factor: int = 50,
    score_weights: "tuple[float, float]" = (0.0, 1.0),
    oxide_threshold: float = 0.0,
    max_n_oxides: Optional[int] = None,
) -> pd.DataFrame:
    """Sample compositions on the ``active_oxides`` simplex, predict ε_r and
    tan δ, filter by the requested ranges, then sort by a combined closeness
    score (higher = better).

    Score components (each normalised to [0, 1]):

    * **ε_r component** – ``1 - |eps_r - mid(eps_r_range)| / half_width``,
      clipped to [0, 1].  1 = perfect centre of the ε_r target window.
    * **tan δ component** – raw ``tan_delta`` value (no normalisation).
      Higher tan δ → higher score → better rank.
      Raw values are used so cross-preset comparisons remain correct.

    ``score = w_eps * eps_component + w_tan * tan_component``
    where ``(w_eps, w_tan) = score_weights`` (default ``(0.0, 1.0)`` —
    ε_r is already enforced as a hard filter so only tan δ drives the score).

    If ``eps_r_range`` is ``None`` the ε_r component is 0.
    If ``tan_delta_range`` is ``None`` the tan δ component is 0.

    ``n_samples`` is the **target number of valid results** to collect after
    the ε_r / tan δ filter.  Sampling proceeds in adaptive batches until this
    target is met or ``n_samples × max_attempts_factor`` total predictions are
    exhausted.  Raise ``max_attempts_factor`` when the ε_r window is very
    narrow and the default budget is not enough.
    """
    extra_props = tuple(extra_props)
    cols_out = (
        list(active_oxides) + ["eps_r", "tan_delta"] + list(extra_props) + ["score"]
    )

    effective_max_k = max_n_oxides if max_n_oxides is not None else len(active_oxides)

    # Pre-compute filter bounds once.
    lo_eps = hi_eps = lo_tan = hi_tan = None
    if eps_r_range is not None:
        lo_eps, hi_eps = float(eps_r_range[0]), float(eps_r_range[1])
    if tan_delta_range is not None:
        lo_tan, hi_tan = float(tan_delta_range[0]), float(tan_delta_range[1])

    # ── Iterative sampling until n_samples *valid* results are collected ──────
    # n_samples is the TARGET valid-result count (after ε_r + tan_δ filter).
    # We sample in batches, predict, keep passing rows, and adapt the next
    # batch size based on the observed acceptance rate.  This way:
    #   - Wide ε_r range (high acceptance) → one batch, no waste.
    #   - Narrow ε_r range (low acceptance) → keeps sampling up to budget.
    # Hard budget: max_attempts_factor × n_samples total predictions.
    target = n_samples
    max_budget = n_samples * max_attempts_factor
    # First batch: sample exactly `target` so that stratified k-quotas are
    # correct when acceptance rate is near 1 (e.g. tests / wide range).
    batch_size = min(target, 20_000)

    valid_wt:  list[pd.DataFrame]  = []
    valid_eps: list[np.ndarray]    = []
    valid_tan: list[np.ndarray]    = []
    n_valid        = 0
    total_sampled  = 0
    current_seed   = seed

    while n_valid < target and total_sampled < max_budget:
        to_sample = min(batch_size, max_budget - total_sampled)
        if to_sample <= 0:
            break

        batch_wt  = _sample_sparse_subsets(
            active_oxides, to_sample, effective_max_k, oxide_threshold,
            current_seed, min_k=3,
        )
        batch_mol = wt_to_mol_frame(batch_wt)
        eps_arr, tan_arr = predictor.batch_eps_tan(batch_mol)
        eps_arr = np.asarray(eps_arr, dtype=float)
        tan_arr = np.asarray(tan_arr, dtype=float)

        mask = np.ones(len(batch_mol), dtype=bool)
        # n_oxides range: only count compositions with 3 ≤ n ≤ effective_max_k
        # toward the target. _sample_sparse_subsets already guarantees this, but
        # the explicit check makes the contract robust to any future change.
        n_ox = (batch_wt > (oxide_threshold if oxide_threshold > 0.0 else 0.0)).sum(axis=1).to_numpy()
        mask &= (n_ox >= 3) & (n_ox <= effective_max_k)
        if lo_eps is not None:
            mask &= np.isfinite(eps_arr) & (eps_arr >= lo_eps) & (eps_arr <= hi_eps)
        if lo_tan is not None:
            mask &= np.isfinite(tan_arr) & (tan_arr >= lo_tan) & (tan_arr <= hi_tan)

        n_pass = int(mask.sum())
        if n_pass > 0:
            valid_wt.append(batch_wt.loc[mask].reset_index(drop=True))
            valid_eps.append(eps_arr[mask])
            valid_tan.append(tan_arr[mask])
            n_valid += n_pass

        total_sampled += to_sample
        if current_seed is not None:
            current_seed += 1

        # Adapt next batch size from observed acceptance rate.
        if n_valid < target and total_sampled > 0:
            rate = n_valid / total_sampled
            if rate > 0:
                needed = target - n_valid
                batch_size = min(max(int(needed / rate * 1.5), 256), 20_000)

    if n_valid == 0:
        return pd.DataFrame(columns=cols_out)

    kept = pd.concat(valid_wt, ignore_index=True).head(target).copy()
    kept["eps_r"]    = np.concatenate(valid_eps)[:target]
    kept["tan_delta"] = np.concatenate(valid_tan)[:target]

    for prop in extra_props:
        kept_mol = wt_to_mol_frame(kept[list(active_oxides)])
        kept[prop] = np.asarray(predictor.batch_property(kept_mol, prop), dtype=float)

    w_eps, w_tan = float(score_weights[0]), float(score_weights[1])

    if eps_r_range is not None:
        lo, hi = float(eps_r_range[0]), float(eps_r_range[1])
        mid  = 0.5 * (lo + hi)
        half = max((hi - lo) / 2.0, 1e-12)
        eps_component = np.clip(
            1.0 - np.abs(kept["eps_r"].to_numpy() - mid) / half, 0.0, 1.0
        )
    else:
        eps_component = np.ones(len(kept))

    if w_tan > 0.0 and len(kept) > 0:
        tan_component = kept["tan_delta"].to_numpy()
    else:
        tan_component = np.zeros(len(kept))

    kept["score"] = w_eps * eps_component + w_tan * tan_component
    kept = kept.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return kept[cols_out]
