"""roadlab_matnav_lib — Roadlab materials navigation library.

Submodules:
  - oxides:    oxide catalog & composition normalization
  - predict:   glasspy-backed property prediction (Tg, P(glass), ε_r, tan δ)
  - recommend: simplex sampling + multi-condition recommender
  - ternary:   python-ternary heatmap helpers
"""

from . import oxides, predict, recommend, screen, ternary, ternary_geom, web

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "oxides",
    "predict",
    "recommend",
    "screen",
    "ternary",
    "ternary_geom",
    "web",
]
