"""Smoke test for T2: glass type selectbox gating.

Tests the UI logic directly (no browser needed) by inspecting
app_glass.py's source structure — ensures:
1. A selectbox with a placeholder option exists before the Run button.
2. The Run button's disabled condition is gated on the selectbox.

These are static analysis / import-time checks that run without a Streamlit
server, so they remain fast in CI.
"""
from __future__ import annotations

import ast
import pathlib

_APP = pathlib.Path(__file__).parent.parent / "app_glass.py"


def _source() -> str:
    return _APP.read_text(encoding="utf-8")


class TestGlassTypeSelectbox:

    def test_selectbox_present(self):
        """app_glass.py must call st.selectbox with 'Glass type' label."""
        src = _source()
        assert 'st.selectbox' in src, "st.selectbox not found in app_glass.py"
        assert 'Glass type' in src, "selectbox label 'Glass type' not found"

    def test_placeholder_option_present(self):
        """The '— 선택 —' placeholder option must be in the selectbox options."""
        src = _source()
        assert '\u2014 \uc120\ud0dd \u2014' in src or '— 선택 —' in src, (
            "placeholder option '— 선택 —' not found in selectbox options"
        )

    def test_aluminosilicate_option_present(self):
        """'Aluminosilicate' must be a selectable option."""
        src = _source()
        assert 'Aluminosilicate' in src, "'Aluminosilicate' option not found"

    def test_run_button_has_disabled_kwarg(self):
        """st.button('Run Search') must include a 'disabled=' keyword argument."""
        src = _source()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match st.button(...)
            if not (isinstance(func, ast.Attribute) and func.attr == 'button'):
                continue
            # Check first positional arg is 'Run Search'
            if not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and first.value == 'Run Search'):
                continue
            # Must have disabled= keyword
            kw_names = [k.arg for k in node.keywords]
            assert 'disabled' in kw_names, (
                "st.button('Run Search') missing 'disabled=' keyword argument"
            )
            return  # found and passed
        raise AssertionError("st.button('Run Search') call not found in app_glass.py")

    def test_run_button_disabled_references_glass_type(self):
        """The disabled= expression must reference the glass_type variable."""
        src = _source()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'button'):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and first.value == 'Run Search'):
                continue
            for kw in node.keywords:
                if kw.arg == 'disabled':
                    # Serialize the disabled expression and check it mentions glass_type
                    disabled_src = ast.unparse(kw.value)
                    assert 'glass_type' in disabled_src, (
                        f"disabled= expression ({disabled_src!r}) does not "
                        "reference 'glass_type'"
                    )
                    return
        raise AssertionError("Could not locate disabled= kwarg on Run Search button")
