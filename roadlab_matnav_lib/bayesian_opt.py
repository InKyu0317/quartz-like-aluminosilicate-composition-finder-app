"""Gaussian-process Bayesian optimisation on the oxide composition simplex.

Public API
----------
run_bo : run BO refinement starting from seed observations (top results from
         the random search) and return all evaluated compositions.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .recommend import _apply_threshold, wt_to_mol_frame

__all__ = ["run_bo"]


# ── acquisition ───────────────────────────────────────────────────────────────

def _ei(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = 0.0,
) -> np.ndarray:
    """Expected Improvement for *minimisation*.

    EI(x) = (y_best - μ(x) - ξ) Φ(z) + σ(x) φ(z)
    where z = (y_best - μ(x) - ξ) / σ(x).
    """
    improvement = y_best - mu - xi
    z = improvement / (sigma + 1e-12)
    return improvement * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)


# ── candidate generation ──────────────────────────────────────────────────────

def _generate_candidates(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    oxide_cols: list[str],
    oxide_threshold: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return candidate compositions (wt%, rows sum to 100) near good observations."""
    K = len(oxide_cols)
    top_idxs = np.argsort(y_obs)[: min(5, len(y_obs))]

    n_exploit = n_candidates * 3 // 4
    n_explore = n_candidates - n_exploit
    parts: list[np.ndarray] = []

    # Exploitation: Dirichlet concentrated around top-5 observations.
    per_best = max(n_exploit // len(top_idxs), 1)
    for bi in top_idxs:
        alpha = np.maximum(X_obs[bi] / 100.0 * 20.0, 0.05)
        batch = rng.dirichlet(alpha, size=per_best) * 100.0
        parts.append(batch)

    # Exploration: uniform inside the bounding box of all observations.
    lo = np.maximum(X_obs.min(axis=0) * 0.5, 0.0)
    hi = np.minimum(X_obs.max(axis=0) * 1.5, 100.0)
    raw = rng.uniform(lo, hi, size=(n_explore, K))
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    parts.append(raw / row_sums * 100.0)

    cands = np.vstack(parts)

    # Apply threshold (vectorised via _apply_threshold).
    cands_df = pd.DataFrame(cands, columns=oxide_cols)
    cands_df = _apply_threshold(cands_df, oxide_threshold)
    if len(cands_df) == 0:
        return np.empty((0, K))
    return cands_df.to_numpy()


# ── main BO loop ──────────────────────────────────────────────────────────────

def run_bo(
    predictor,
    oxide_cols: Sequence[str],
    seed_df: pd.DataFrame,
    *,
    n_iter: int = 30,
    n_candidates: int = 5_000,
    eps_r_range: Optional[tuple[float, float]] = None,
    oxide_threshold: float = 1.0,
    rng_seed: int = 42,
    top_m_candidates: int = 20,
    callback: Optional[Callable[[int, int, float], None]] = None,
) -> pd.DataFrame:
    """Run GP-BO refinement starting from *seed_df* observations.

    Parameters
    ----------
    predictor:
        GlassPredictor instance (already warmed up).
    oxide_cols:
        Ordered list of oxide column names to optimise over.
    seed_df:
        DataFrame with columns ``oxide_cols`` (wt%), ``eps_r``, ``tan_delta``.
        Typically the top-N rows from the random search.
    n_iter:
        Number of BO iterations (one GlassNet call per iteration).
    n_candidates:
        Random candidates to evaluate EI against per iteration.
    eps_r_range:
        If given, proposed compositions must satisfy this ε_r constraint.
    oxide_threshold:
        Minor-oxide zeroing threshold (wt%).
    rng_seed:
        Base RNG seed.
    top_m_candidates:
        How many top-EI candidates to evaluate before giving up in a BO step
        (useful when eps_r constraint rejects many candidates).
    callback:
        Optional callable(iteration, n_iter, best_tan_so_far) called after
        each iteration — useful for updating a Streamlit progress bar.

    Returns
    -------
    DataFrame with columns ``oxide_cols + ['eps_r', 'tan_delta', 'source']``,
    sorted by *tan_delta* ascending.  ``source`` is ``'seed'`` for initial
    observations and ``'bo'`` for BO-discovered compositions.
    """
    oxide_cols = list(oxide_cols)
    K = len(oxide_cols)
    rng = np.random.default_rng(rng_seed)

    # Observations accumulator — starts from seed.
    X_obs = seed_df[oxide_cols].fillna(0.0).to_numpy(dtype=float)
    y_obs = seed_df["tan_delta"].to_numpy(dtype=float)
    eps_obs = seed_df["eps_r"].to_numpy(dtype=float)

    # GP kernel: ARD Matern(ν=2.5) + small noise.
    kernel = (
        Matern(
            nu=2.5,
            length_scale=np.ones(K),
            length_scale_bounds=(1e-2, 1e6),  # wide bounds avoid ConvergenceWarning
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    )
    scaler = StandardScaler()

    bo_X: list[np.ndarray] = []
    bo_eps: list[float] = []
    bo_tan: list[float] = []
    bo_iter_nums: list[int] = []

    for i in range(n_iter):
        # ── 1. Fit GP on current observations ────────────────────────────────
        X_scaled = scaler.fit_transform(X_obs)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=rng_seed + i,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_scaled, y_obs)

        # ── 2. Generate candidates ────────────────────────────────────────────
        cands = _generate_candidates(
            X_obs, y_obs, oxide_cols, oxide_threshold, n_candidates, rng
        )
        if len(cands) == 0:
            if callback:
                callback(i + 1, n_iter, float(y_obs.min()))
            continue

        # ── 3. Evaluate EI ────────────────────────────────────────────────────
        cands_scaled = scaler.transform(cands)
        mu, sigma = gp.predict(cands_scaled, return_std=True)
        y_best = float(y_obs.min())
        ei_vals = _ei(mu, sigma, y_best)

        # ── 4. Pick best EI candidate(s), evaluate with GlassNet ─────────────
        n_top = min(top_m_candidates, len(ei_vals))
        top_idx = np.argpartition(ei_vals, -n_top)[-n_top:]
        top_idx = top_idx[np.argsort(ei_vals[top_idx])[::-1]]  # desc by EI

        evaluated = False
        for idx in top_idx:
            new_wt = pd.DataFrame(cands[idx : idx + 1], columns=oxide_cols)
            new_mol = wt_to_mol_frame(new_wt.fillna(0.0))
            eps_arr, tan_arr = predictor.batch_eps_tan(new_mol)
            eps_val = float(np.asarray(eps_arr, dtype=float)[0])
            tan_val = float(np.asarray(tan_arr, dtype=float)[0])

            if not (np.isfinite(eps_val) and np.isfinite(tan_val)):
                continue
            if eps_r_range is not None:
                if not (eps_r_range[0] <= eps_val <= eps_r_range[1]):
                    continue

            # Accept this point.
            X_obs = np.vstack([X_obs, cands[idx]])
            y_obs = np.append(y_obs, tan_val)
            eps_obs = np.append(eps_obs, eps_val)
            bo_X.append(cands[idx].copy())
            bo_eps.append(eps_val)
            bo_tan.append(tan_val)
            bo_iter_nums.append(i + 1)
            evaluated = True
            break

        if callback:
            callback(i + 1, n_iter, float(y_obs.min()))

    # ── Build result DataFrame ────────────────────────────────────────────────
    seed_out = seed_df[oxide_cols + ["eps_r", "tan_delta"]].copy()
    seed_out["source"] = "seed"

    if not bo_X:
        return seed_out.sort_values("tan_delta").reset_index(drop=True)

    bo_df = pd.DataFrame(bo_X, columns=oxide_cols)
    bo_df["eps_r"] = bo_eps
    bo_df["tan_delta"] = bo_tan
    bo_df["bo_iter"] = bo_iter_nums
    bo_df["source"] = "bo"

    return (
        pd.concat([seed_out, bo_df], ignore_index=True)
        .sort_values("tan_delta")
        .reset_index(drop=True)
    )
