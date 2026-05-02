"""Glass property prediction — thin wrappers over `glasspy.predict`.

Provides :class:`GlassPredictor` exposing two operations:

- :meth:`GlassPredictor.tg`                  — glass transition temperature (K)
- :meth:`GlassPredictor.glass_probability`   — VITRIFY P(glass) ∈ [0, 1]

All inputs are validated through :func:`roadlab_matnav_lib.oxides.normalize`
so that compositions sum to 100 mol% and contain only catalog-known oxides.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Union

import numpy as np
import pandas as pd

from glasspy.predict import GlassNet, VITRIFY

from . import oxides

__all__ = ["GlassPredictor"]

CompositionLike = Union[Mapping[str, float], pd.DataFrame, Iterable[Mapping[str, float]]]


class GlassPredictor:
    """Lazy-loading wrapper around glasspy's GlassNet and VITRIFY models."""

    def __init__(self, *, vitrify_model: str = "CHEM") -> None:
        self._vitrify_model = vitrify_model
        self._glassnet: "GlassNet | None" = None
        self._vitrify: "VITRIFY | None" = None

    # -- model accessors (lazy) -------------------------------------------------

    def _get_glassnet(self) -> GlassNet:
        if self._glassnet is None:
            self._glassnet = GlassNet()
        return self._glassnet

    def _get_vitrify(self) -> VITRIFY:
        if self._vitrify is None:
            self._vitrify = VITRIFY(model=self._vitrify_model)
        return self._vitrify

    # -- single-composition API -------------------------------------------------

    def tg(self, composition: Mapping[str, float]) -> float:
        """Predict the glass transition temperature (K) for a single composition."""
        df_in = self._compositions_to_frame([composition])
        df = self._get_glassnet().predict(df_in)
        return float(df["Tg"].iloc[0])

    def glass_probability(self, composition: Mapping[str, float]) -> float:
        """Predict P(glass) ∈ [0, 1] for a single composition (VITRIFY)."""
        comp = oxides.normalize(composition)
        df = pd.DataFrame([comp]).fillna(0.0)
        proba = self._get_vitrify().predict_proba_glass(df)
        return float(np.asarray(proba).ravel()[0])

    # -- batch API --------------------------------------------------------------

    def batch_glass_probability(self, compositions: CompositionLike) -> np.ndarray:
        """Predict P(glass) for many compositions in a single model call.

        Accepts a list of dicts or a pandas DataFrame whose columns are oxide
        formulas. Each row is normalized to sum to 100 before inference.
        """
        df = self._compositions_to_frame(compositions)
        proba = self._get_vitrify().predict_proba_glass(df)
        return np.asarray(proba).ravel()

    def batch_property(
        self, compositions: CompositionLike, prop: str
    ) -> np.ndarray:
        """Predict any GlassNet property column for many compositions in one call.

        ``prop`` must be one of GlassNet's output columns (e.g. ``'Tg'``,
        ``'Tmelt'``, ``'CTEbelowTg'``, ``'YoungModulus'`` …). Raises
        :class:`KeyError` if the column is not produced by the model.
        """
        df_in = self._compositions_to_frame(compositions)
        df_out = self._get_glassnet().predict(df_in)
        if prop not in df_out.columns:
            raise KeyError(
                f"property {prop!r} is not in GlassNet output columns"
            )
        return df_out[prop].to_numpy()

    def batch_dielectric_constant(self, compositions: CompositionLike) -> np.ndarray:
        """Predict the dielectric constant ε_r (GlassNet ``Permittivity``)."""
        return self.batch_property(compositions, "Permittivity")

    def batch_dielectric_loss(self, compositions: CompositionLike) -> np.ndarray:
        """Predict the dielectric loss tan δ.

        GlassNet's ``TangentOfLossAngle`` head outputs ``log10(tan δ)``
        (see ``glasspy/data/translators.py``). This method decodes it and
        returns the actual ``tan δ`` value, so callers work in physical units.
        """
        raw = self.batch_property(compositions, "TangentOfLossAngle")
        return np.power(10.0, raw, where=np.isfinite(raw),
                        out=np.full_like(raw, np.nan, dtype=float))

    def batch_in_range(
        self,
        compositions: CompositionLike,
        *,
        eps_r_range: "tuple[float, float] | None" = None,
        tan_delta_range: "tuple[float, float] | None" = None,
    ) -> np.ndarray:
        """Return a boolean mask for compositions whose predicted dielectric
        properties fall within the requested closed intervals.

        - ``None`` ranges are skipped (not enforced).
        - ``NaN`` predictions yield ``False`` (safe path).
        - Issues a single GlassNet call regardless of how many conditions are
          requested.
        """
        df_in = self._compositions_to_frame(compositions)
        df_out = self._get_glassnet().predict(df_in)
        n = len(df_in)
        mask = np.ones(n, dtype=bool)

        for col, rng in (
            ("Permittivity", eps_r_range),
            ("TangentOfLossAngle", tan_delta_range),
        ):
            if rng is None:
                continue
            if col not in df_out.columns:
                raise KeyError(f"property {col!r} is not in GlassNet output columns")
            lo, hi = float(rng[0]), float(rng[1])
            vals = df_out[col].to_numpy(dtype=float)
            # GlassNet outputs ``TangentOfLossAngle`` as log10(tan δ); decode
            # so the user-facing ``tan_delta_range`` is in physical units.
            if col == "TangentOfLossAngle":
                vals = np.power(10.0, vals, where=np.isfinite(vals),
                                out=np.full_like(vals, np.nan))
            ok = (vals >= lo) & (vals <= hi) & np.isfinite(vals)
            mask &= ok

        return mask

    def batch_thermal(self, compositions: CompositionLike) -> pd.DataFrame:
        """Predict thermal/structural-stability properties in a single GlassNet
        call.

        Returns a DataFrame with columns:

        - ``Tg`` — glass-transition temperature (K)
        - ``Tx`` — crystallization onset (K, GlassNet ``CrystallizationOnset``)
        - ``Tliquidus`` — liquidus temperature (K)
        - ``CTE_per_K`` — coefficient of thermal expansion below Tg in 1/K
          (decoded as ``10 ** CTEbelowTg``)
        - ``delta_T`` — ``Tx - Tg`` (proxy for glass-forming ability)

        NaN inputs propagate to NaN outputs without raising.
        """
        df_in = self._compositions_to_frame(compositions)
        df_out = self._get_glassnet().predict(df_in)

        needed = ["Tg", "CrystallizationOnset", "Tliquidus", "CTEbelowTg"]
        for col in needed:
            if col not in df_out.columns:
                raise KeyError(f"property {col!r} is not in GlassNet output columns")

        tg = df_out["Tg"].to_numpy(dtype=float)
        tx = df_out["CrystallizationOnset"].to_numpy(dtype=float)
        tliq = df_out["Tliquidus"].to_numpy(dtype=float)
        cte_log = df_out["CTEbelowTg"].to_numpy(dtype=float)
        cte = np.power(10.0, cte_log, where=np.isfinite(cte_log), out=np.full_like(cte_log, np.nan))
        delta = tx - tg

        return pd.DataFrame({
            "Tg": tg,
            "Tx": tx,
            "Tliquidus": tliq,
            "CTE_per_K": cte,
            "delta_T": delta,
        })

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _compositions_to_frame(compositions: CompositionLike) -> pd.DataFrame:
        if isinstance(compositions, pd.DataFrame):
            rows: List[Mapping[str, float]] = compositions.to_dict(orient="records")
        else:
            rows = list(compositions)  # type: ignore[arg-type]
        normalized = [oxides.normalize(r) for r in rows]
        return pd.DataFrame(normalized).fillna(0.0)
