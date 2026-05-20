"""T5: Sidebar widget order test.

Verifies that, within the with st.sidebar: block, the st.* calls appear
in the specified order. We use line numbers from the AST to compare.

Expected order:
  glass_type (selectbox) < alkali (radio) < ε_r (slider) < tan_target (slider)
  < p_glass (slider) < tg_range (slider) < cte_range (slider)
  < sio2 (slider) < max_n_oxides (slider) < top_n (slider)
  < al2o3_second (checkbox) < run (button)
"""
from __future__ import annotations

import ast
import pathlib
from typing import Optional

_APP = pathlib.Path(__file__).parent.parent / "app_glass.py"
_SRC = _APP.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _line_of_assign(target_name: str) -> Optional[int]:
    """Return the line number of the first assignment to target_name (handles tuples)."""
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            # Direct name assignment: x = ...
            if isinstance(t, ast.Name) and t.id == target_name:
                return node.lineno
            # Tuple unpacking: a, b = ...
            if isinstance(t, ast.Tuple):
                for elt in t.elts:
                    if isinstance(elt, ast.Name) and elt.id == target_name:
                        return node.lineno
    return None


def _line_of_button(label: str) -> Optional[int]:
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "button"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == label:
            return node.lineno
    return None


def _assert_order(a_name, a_line, b_name, b_line):
    assert a_line is not None, f"Could not find '{a_name}' assignment in app_glass.py"
    assert b_line is not None, f"Could not find '{b_name}' assignment in app_glass.py"
    assert a_line < b_line, (
        f"Expected {a_name} (line {a_line}) to appear before {b_name} (line {b_line})"
    )


class TestSidebarOrder:

    def test_glass_type_before_alkali(self):
        _assert_order("glass_type", _line_of_assign("glass_type"),
                      "alkali_mode", _line_of_assign("alkali_mode"))

    def test_alkali_before_eps(self):
        _assert_order("alkali_mode", _line_of_assign("alkali_mode"),
                      "eps_min", _line_of_assign("eps_min"))

    def test_eps_before_tan_target(self):
        _assert_order("eps_min", _line_of_assign("eps_min"),
                      "tan_target", _line_of_assign("tan_target"))

    def test_tan_target_before_p_glass(self):
        _assert_order("tan_target", _line_of_assign("tan_target"),
                      "p_glass_min", _line_of_assign("p_glass_min"))

    def test_p_glass_before_tg_range(self):
        _assert_order("p_glass_min", _line_of_assign("p_glass_min"),
                      "tg_range", _line_of_assign("tg_range"))

    def test_tg_range_before_cte_range(self):
        _assert_order("tg_range", _line_of_assign("tg_range"),
                      "cte_range", _line_of_assign("cte_range"))

    def test_cte_range_before_sio2(self):
        _assert_order("cte_range", _line_of_assign("cte_range"),
                      "sio2_min", _line_of_assign("sio2_min"))

    def test_sio2_before_max_n_oxides(self):
        _assert_order("sio2_min", _line_of_assign("sio2_min"),
                      "max_n_oxides", _line_of_assign("max_n_oxides"))

    def test_max_n_oxides_before_top_n(self):
        _assert_order("max_n_oxides", _line_of_assign("max_n_oxides"),
                      "top_n", _line_of_assign("top_n"))

    def test_top_n_before_al2o3_second(self):
        _assert_order("top_n", _line_of_assign("top_n"),
                      "al2o3_second", _line_of_assign("al2o3_second"))

    def test_al2o3_second_before_run_button(self):
        run_line = _line_of_button("Run Search")
        al2o3_line = _line_of_assign("al2o3_second")
        _assert_order("al2o3_second", al2o3_line, "run_button", run_line)
