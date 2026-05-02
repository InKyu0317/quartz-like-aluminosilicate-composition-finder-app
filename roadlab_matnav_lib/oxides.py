"""Oxide catalog, formula parsing, and composition normalization.

Catalog is curated for glass / ceramics work. Each entry stores the chemical
formula, molecular weight (computed from `glasspy.chemistry.elementmass`),
elemental composition, and a coarse role classification:

- ``former``       : primary network former (SiO2, B2O3, P2O5, GeO2, ...)
- ``modifier``     : network modifier (alkali / alkaline earth oxides)
- ``intermediate`` : conditional network former (Al2O3, TiO2, ZnO, ...)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping

from glasspy.chemistry import elementmass as _ELEMENT_MASS

__all__ = [
    "OxideInfo",
    "list_supported",
    "info",
    "parse_formula",
    "normalize",
    "wt_to_mol",
    "mol_to_wt",
]


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_VALID = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula like 'Al2O3' into ``{'Al': 2, 'O': 3}``.

    Only flat formulas (no parentheses, no hydrates) are supported.
    """
    if not isinstance(formula, str) or not _FORMULA_VALID.match(formula):
        raise ValueError(f"invalid chemical formula: {formula!r}")
    counts: Dict[str, int] = {}
    for element, count in _FORMULA_TOKEN.findall(formula):
        if not element:
            continue
        if element not in _ELEMENT_MASS:
            raise ValueError(f"unknown element {element!r} in formula {formula!r}")
        counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    if not counts:
        raise ValueError(f"empty formula: {formula!r}")
    return counts


def _mw(elements: Mapping[str, int]) -> float:
    return sum(_ELEMENT_MASS[el] * n for el, n in elements.items())


@dataclass(frozen=True)
class OxideInfo:
    """Metadata for a single oxide entry."""

    formula: str
    mw: float
    elements: Mapping[str, int]
    role: str  # 'former' | 'modifier' | 'intermediate'


def _make(formula: str, role: str) -> OxideInfo:
    els = parse_formula(formula)
    return OxideInfo(
        formula=formula,
        mw=_mw(els),
        elements=MappingProxyType(els),
        role=role,
    )


# Curated oxide catalog. Add new entries here.
_CATALOG: Dict[str, OxideInfo] = {
    o.formula: o
    for o in (
        # Network formers
        _make("SiO2", "former"),
        _make("B2O3", "former"),
        _make("P2O5", "former"),
        _make("GeO2", "former"),
        _make("As2O3", "former"),
        _make("As2O5", "former"),
        _make("V2O5", "former"),
        # Intermediates
        _make("Al2O3", "intermediate"),
        _make("TiO2", "intermediate"),
        _make("ZrO2", "intermediate"),
        _make("ZnO", "intermediate"),
        _make("PbO", "intermediate"),
        _make("Bi2O3", "intermediate"),
        _make("Sb2O3", "intermediate"),
        _make("Ga2O3", "intermediate"),
        _make("In2O3", "intermediate"),
        _make("SnO2", "intermediate"),
        _make("Ta2O5", "intermediate"),
        _make("Nb2O5", "intermediate"),
        # Alkali modifiers
        _make("Li2O", "modifier"),
        _make("Na2O", "modifier"),
        _make("K2O", "modifier"),
        _make("Rb2O", "modifier"),
        _make("Cs2O", "modifier"),
        # Alkaline earth modifiers
        _make("BeO", "modifier"),
        _make("MgO", "modifier"),
        _make("CaO", "modifier"),
        _make("SrO", "modifier"),
        _make("BaO", "modifier"),
        # Transition metal / other modifiers
        _make("FeO", "modifier"),
        _make("Fe2O3", "modifier"),
        _make("MnO", "modifier"),
        _make("MnO2", "modifier"),
        _make("CoO", "modifier"),
        _make("NiO", "modifier"),
        _make("CuO", "modifier"),
        _make("Cu2O", "modifier"),
        _make("Cr2O3", "modifier"),
        _make("MoO3", "modifier"),
        _make("WO3", "modifier"),
        _make("La2O3", "modifier"),
        _make("Y2O3", "modifier"),
        _make("CeO2", "modifier"),
    )
}


def list_supported() -> List[str]:
    """Return the sorted list of supported oxide formulas."""
    return sorted(_CATALOG)


def info(formula: str) -> OxideInfo:
    """Return :class:`OxideInfo` for ``formula``.

    Raises :class:`KeyError` if the oxide is not in the catalog.
    """
    try:
        return _CATALOG[formula]
    except KeyError:
        raise KeyError(f"unsupported oxide: {formula!r}") from None


def normalize(composition: Mapping[str, float]) -> Dict[str, float]:
    """Validate and rescale a composition dict so values sum to 100.

    - Keys must be in :func:`list_supported`.
    - Values must be non-negative and finite.
    - Zero-valued entries are dropped from the output.
    - The total must be > 0.
    """
    if not isinstance(composition, Mapping):
        raise TypeError(f"composition must be a mapping, got {type(composition).__name__}")
    cleaned: Dict[str, float] = {}
    for key, value in composition.items():
        if key not in _CATALOG:
            raise KeyError(f"unsupported oxide: {key!r}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value for {key!r}: {value!r}")
        if v < 0:
            raise ValueError(f"negative value for {key!r}: {value!r}")
        if v > 0:
            cleaned[key] = v
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("composition total must be > 0")
    scale = 100.0 / total
    return {k: v * scale for k, v in cleaned.items()}


def wt_to_mol(composition: Mapping[str, float]) -> Dict[str, float]:
    """Convert a weight-percent composition to mol-percent.

    Each oxide's weight is divided by its molecular weight, then the result is
    normalised to sum to 100.  Only positive values contribute; zero-valued
    entries are dropped from the output.

    Raises :class:`KeyError` for unknown oxides and :class:`ValueError` for
    non-positive or non-finite values (same rules as :func:`normalize`).
    """
    molar: Dict[str, float] = {}
    for key, value in composition.items():
        if key not in _CATALOG:
            raise KeyError(f"unsupported oxide: {key!r}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value for {key!r}: {value!r}")
        if v < 0:
            raise ValueError(f"negative value for {key!r}: {value!r}")
        if v > 0:
            molar[key] = v / _CATALOG[key].mw
    total = sum(molar.values())
    if total <= 0:
        raise ValueError("composition total must be > 0")
    return {k: v / total * 100.0 for k, v in molar.items()}


def mol_to_wt(composition: Mapping[str, float]) -> Dict[str, float]:
    """Convert a mol-percent composition to weight-percent.

    Each oxide's molar fraction is multiplied by its molecular weight, then the
    result is normalised to sum to 100.  Only positive values contribute;
    zero-valued entries are dropped from the output.
    """
    wt: Dict[str, float] = {}
    for key, value in composition.items():
        if key not in _CATALOG:
            raise KeyError(f"unsupported oxide: {key!r}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value for {key!r}: {value!r}")
        if v < 0:
            raise ValueError(f"negative value for {key!r}: {value!r}")
        if v > 0:
            wt[key] = v * _CATALOG[key].mw
    total = sum(wt.values())
    if total <= 0:
        raise ValueError("composition total must be > 0")
    return {k: v / total * 100.0 for k, v in wt.items()}
