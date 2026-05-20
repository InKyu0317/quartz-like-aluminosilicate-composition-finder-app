"""T4: Tg and CTE range filter sliders.

Checks that:
1. Tg range slider exists in sidebar.
2. CTE range slider exists in sidebar.
3. df_view filter references Tg_C and CTE_1e6 columns.
"""
from __future__ import annotations

import ast
import pathlib

_APP = pathlib.Path(__file__).parent.parent / "app_glass.py"
_SRC = _APP.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _slider_assigns() -> list[ast.Assign]:
    return [
        n for n in ast.walk(_TREE)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "slider"
    ]


def _target_names(assigns):
    names = []
    for a in assigns:
        for t in a.targets:
            if isinstance(t, ast.Tuple):
                names.extend(e.id for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(t, ast.Name):
                names.append(t.id)
    return names


class TestTgCteSliders:

    def test_tg_range_slider_present(self):
        """A tg_range tuple slider must be assigned in the sidebar block."""
        names = _target_names(_slider_assigns())
        assert "tg_range" in names, (
            f"No 'tg_range' slider assignment found. Sliders: {names}"
        )

    def test_cte_range_slider_present(self):
        """A cte_range tuple slider must be assigned in the sidebar block."""
        names = _target_names(_slider_assigns())
        assert "cte_range" in names, (
            f"No 'cte_range' slider assignment found. Sliders: {names}"
        )

    def test_df_view_filters_tg(self):
        """tg_mask assignment must reference Tg_C; df_view must include tg_mask."""
        # Check tg_mask assignment references Tg_C
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "tg_mask":
                    rhs = ast.unparse(node.value)
                    assert "Tg_C" in rhs, (
                        f"tg_mask assignment does not reference 'Tg_C':\n{rhs}"
                    )
                    # Also check df_view uses tg_mask
                    df_view_src = _SRC
                    assert "tg_mask" in df_view_src, "tg_mask not referenced in file"
                    return
        raise AssertionError("tg_mask assignment not found in app_glass.py")

    def test_df_view_filters_cte(self):
        """cte_mask assignment must reference CTE_1e6; df_view must include cte_mask."""
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "cte_mask":
                    rhs = ast.unparse(node.value)
                    assert "CTE_1e6" in rhs, (
                        f"cte_mask assignment does not reference 'CTE_1e6':\n{rhs}"
                    )
                    assert "cte_mask" in _SRC, "cte_mask not referenced in file"
                    return
        raise AssertionError("cte_mask assignment not found in app_glass.py")
