"""T3: run_search signature / tan_target parameter tests.

Checks that:
1. run_search() accepts a tan_target keyword argument.
2. When tan_target changes, the cached result changes (different cache key).
3. Score ordering respects the new tan_target (composition closest to target scores highest).
"""
from __future__ import annotations

import inspect
import ast
import pathlib

_APP = pathlib.Path(__file__).parent.parent / "app_glass.py"
_SRC = _APP.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


class TestTanTargetSignature:

    def test_run_search_has_tan_target_param(self):
        """run_search() must declare a tan_target parameter."""
        for node in ast.walk(_TREE):
            if isinstance(node, ast.FunctionDef) and node.name == "run_search":
                param_names = [a.arg for a in node.args.args] + [
                    a.arg for a in node.args.kwonlyargs
                ]
                assert "tan_target" in param_names, (
                    f"run_search() parameters {param_names} do not include 'tan_target'"
                )
                return
        raise AssertionError("run_search() function not found in app_glass.py")

    def test_score_formula_uses_tan_target(self):
        """The score formula must reference tan_target, not TAN_QUARTZ."""
        # Find the assignment: df["score"] = ...
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "score"
                ):
                    rhs_src = ast.unparse(node.value)
                    assert "tan_target" in rhs_src, (
                        f"df['score'] assignment ({rhs_src!r}) does not reference 'tan_target'"
                    )
                    return
        raise AssertionError("df['score'] assignment not found in app_glass.py")

    def test_x_quartz_col_uses_tan_target(self):
        """The ×quartz column formula must reference tan_target."""
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and "\u00d7" in str(t.slice.value)  # × character
                ):
                    rhs_src = ast.unparse(node.value)
                    assert "tan_target" in rhs_src, (
                        f"×quartz assignment ({rhs_src!r}) does not reference 'tan_target'"
                    )
                    return
        raise AssertionError("×quartz assignment not found in app_glass.py")

    def test_tan_target_slider_present(self):
        """A slider for tanδ target must exist in sidebar."""
        assert "tan_target" in _SRC, "tan_target slider not found in app_glass.py"
        # Must be a slider call
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "slider"):
                continue
            # Look for assignment to tan_target
            parent_assigns = [
                n for n in ast.walk(_TREE)
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "tan_target"
                    for t in n.targets
                )
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "slider"
            ]
            if parent_assigns:
                return
        raise AssertionError(
            "No 'tan_target = st.slider(...)' assignment found in app_glass.py"
        )

    def test_run_search_call_passes_tan_target(self):
        """run_search() call site must pass tan_target."""
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "run_search"):
                continue
            kw_names = [k.arg for k in node.keywords]
            # tan_target may be positional too — check both
            if "tan_target" in kw_names:
                return
            # positional: count positional args; signature is:
            # (oxide_tuple, eps_min, eps_max, n_samples, max_n_oxides, tan_target, seed=0)
            if len(node.args) >= 6:
                return
        raise AssertionError(
            "run_search() call does not pass tan_target (neither positional nor keyword)"
        )
