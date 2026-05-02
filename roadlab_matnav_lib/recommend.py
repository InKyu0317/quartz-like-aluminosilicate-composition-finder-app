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

    ``max_attempts_factor`` is forwarded to :func:`sample_simplex`; raise it
    when ``fixed`` constraints are tight (e.g. ``SiO2 >= 50`` on an 8-oxide
    simplex) to avoid the rejection-sampling budget being exceeded.
    """
    extra_props = tuple(extra_props)
    cols_out = (
        list(active_oxides) + ["eps_r", "tan_delta"] + list(extra_props) + ["score"]
    )

    # sample_simplex returns compositions in wt%.  GlassNet expects mol%, so we
    # convert each row before any predictor call.  The output DataFrame keeps
    # the original wt% values so callers see weight-percent compositions.
    samples_wt = sample_simplex(
        active_oxides, n_samples,
        fixed=fixed, seed=seed,
        max_attempts_factor=max_attempts_factor,
    )
    samples_mol = wt_to_mol_frame(samples_wt)

    eps = np.asarray(predictor.batch_dielectric_constant(samples_mol), dtype=float)
    tan = np.asarray(predictor.batch_dielectric_loss(samples_mol), dtype=float)
    # Build the filter mask directly from already-computed arrays — avoids
    # extra GlassNet.predict() calls that batch_in_range() would trigger.
    mask = np.ones(len(samples_mol), dtype=bool)
    if eps_r_range is not None:
        lo_eps, hi_eps = float(eps_r_range[0]), float(eps_r_range[1])
        mask &= np.isfinite(eps) & (eps >= lo_eps) & (eps <= hi_eps)
    if tan_delta_range is not None:
        lo_tan, hi_tan = float(tan_delta_range[0]), float(tan_delta_range[1])
        mask &= np.isfinite(tan) & (tan >= lo_tan) & (tan <= hi_tan)
    mask = np.asarray(mask, dtype=bool)

    kept = samples_wt.loc[mask].reset_index(drop=True).copy()  # wt% output
    if kept.empty:
        return pd.DataFrame(columns=cols_out)

    kept["eps_r"] = eps[mask]
    kept["tan_delta"] = tan[mask]

    for prop in extra_props:
        vals = np.asarray(predictor.batch_property(samples_mol, prop), dtype=float)
        kept[prop] = vals[mask]

    w_eps, w_tan = float(score_weights[0]), float(score_weights[1])

    if eps_r_range is not None:
        lo, hi = float(eps_r_range[0]), float(eps_r_range[1])
        mid = 0.5 * (lo + hi)
        half = max((hi - lo) / 2.0, 1e-12)
        eps_component = np.clip(
            1.0 - np.abs(kept["eps_r"].to_numpy() - mid) / half, 0.0, 1.0
        )
    else:
        eps_component = np.ones(len(kept))

    if w_tan > 0.0 and len(kept) > 0:
        # Use raw tan_delta as the component (no per-call normalisation).
        # Normalising by per-preset max causes cross-preset rank inversion:
        # e.g. alkali_free max=0.003 → score=1.0 beats abs tan=0.011 →
        # score=0.22, even though 0.011 > 0.003.
        # With raw values, score = tan_delta and the global ranking is correct.
        tan_component = kept["tan_delta"].to_numpy()
    else:
        tan_component = np.zeros(len(kept))

    kept["score"] = w_eps * eps_component + w_tan * tan_component

    kept = kept.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return kept[cols_out]
