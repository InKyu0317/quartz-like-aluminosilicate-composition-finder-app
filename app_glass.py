"""Streamlit dashboard — alkali-free aluminosilicate composition finder.

Run:
    streamlit run app_glass.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from roadlab_matnav_lib.predict import GlassPredictor
import roadlab_matnav_lib as rml

# ── constants ────────────────────────────────────────────────────────────────
TAN_QUARTZ      = 0.000198
OXIDE_THRESHOLD = 1.0
FIXED = {'SiO2': (40.0, 85.0)}

ALKALI_FREE_OXIDES = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'ZrO2',
    'B2O3', 'La2O3', 'Y2O3',
]
WITH_ALKALI_OXIDES = ALKALI_FREE_OXIDES + ['Na2O', 'K2O', 'Li2O']

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Glass Composition Finder", layout="wide")
st.title("Glass Composition Finder")
st.caption(f"GlassNet · quartz reference: tan\u03b4 = {TAN_QUARTZ:.6f}, \u03b5_r = 3.77")

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Parameters")

    alkali_mode = st.radio("Alkali content", ["Alkali-free", "Include alkali"])
    active_oxides = ALKALI_FREE_OXIDES if alkali_mode == "Alkali-free" else WITH_ALKALI_OXIDES

    eps_min, eps_max = st.slider(
        "\u03b5_r range", min_value=3.0, max_value=15.0, value=(3.77, 10.0), step=0.1
    )
    max_n_oxides = st.slider(
        "Max oxide count", min_value=2, max_value=len(active_oxides), value=len(active_oxides)
    )
    n_samples = st.select_slider(
        "Sample count", options=[10_000, 30_000, 50_000, 100_000], value=50_000
    )
    top_n = st.slider("Rows to display", min_value=10, max_value=200, value=30)
    run = st.button("Run Search", type="primary")

# ── model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="GlassNet \ubaa8\ub378 \ub85c\ub529 \uc911\u2026")
def load_predictor():
    return GlassPredictor()

# ── search (cached by params) ─────────────────────────────────────────────────
@st.cache_data(show_spinner="\uc870\uc131 \uc0d8\ud50c\ub9c1 \uc911\u2026", max_entries=8)
def run_search(oxide_tuple, eps_min, eps_max, n_samples, seed=0):
    predictor = load_predictor()
    oxides = list(oxide_tuple)
    df = rml.recommend.recommend(
        predictor,
        active_oxides=oxides,
        eps_r_range=(eps_min, eps_max),
        tan_delta_range=(0.0, 1.0),
        n_samples=n_samples,
        fixed=FIXED,
        seed=seed,
        max_attempts_factor=1000,
    ).fillna(0.0)

    oxide_cols = [c for c in oxides if c in df.columns]
    df["score"]    = 1.0 / (1.0 + np.abs(df["tan_delta"] - TAN_QUARTZ) / TAN_QUARTZ)
    df["n_oxides"] = (df[oxide_cols] > OXIDE_THRESHOLD).sum(axis=1).astype(int)
    df["\u00d7quartz"] = (df["tan_delta"] / TAN_QUARTZ).round(2)
    df = df.sort_values(["score", "n_oxides"], ascending=[False, True]).reset_index(drop=True)
    df.index += 1  # 1-based rank
    return df, oxide_cols

# ── run only when button clicked ──────────────────────────────────────────────
if run:
    df, oxide_cols = run_search(tuple(active_oxides), eps_min, eps_max, n_samples)
    st.session_state["df"] = df
    st.session_state["oxide_cols"] = oxide_cols

if "df" not in st.session_state:
    st.info("\uc67c\ucabd \uc0ac\uc774\ub4dc\ubc14\uc5d0\uc11c \ud30c\ub77c\ubbf8\ud130\ub97c \uc124\uc815\ud558\uace0 **Run Search** \ub97c \ub204\ub974\uc138\uc694.")
    st.stop()

df = st.session_state["df"]
oxide_cols = st.session_state["oxide_cols"]

# ── apply n_oxides filter ─────────────────────────────────────────────────────
df_view = df[df["n_oxides"] <= max_n_oxides].head(top_n)

# ── metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total candidates", f"{len(df):,}")
col2.metric("Shown (after filter)", f"{len(df_view):,}")
best = df_view.iloc[0] if len(df_view) else df.iloc[0]
col3.metric("Best \u03b5_r", f"{best['eps_r']:.3f}")
col4.metric("Best tan\u03b4 / quartz", f"{best['\u00d7quartz']:.2f}\u00d7")

# ── table ─────────────────────────────────────────────────────────────────────
display_cols = ["eps_r", "tan_delta", "\u00d7quartz", "score", "n_oxides"] + [
    c for c in oxide_cols if c in df_view.columns
]
st.dataframe(
    df_view[display_cols].style.format({
        "eps_r":        "{:.3f}",
        "tan_delta":    "{:.6f}",
        "\u00d7quartz": "{:.2f}\u00d7",
        "score":        "{:.4f}",
        **{c: "{:.1f}" for c in oxide_cols},
    }).background_gradient(subset=["score"], cmap="RdYlGn")
     .background_gradient(subset=["tan_delta"], cmap="RdYlGn_r"),
    width="stretch",
    height=600,
)

# ── detail for selected row ───────────────────────────────────────────────────
st.divider()
st.subheader("Composition detail")
rank = st.number_input("Rank to inspect", min_value=1, max_value=max(len(df_view), 1), value=1, step=1)
if rank <= len(df_view):
    row = df_view.iloc[rank - 1]
    present = {c: row[c] for c in oxide_cols if row.get(c, 0) > OXIDE_THRESHOLD}
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**\u03b5_r** = {row['eps_r']:.4f}")
        st.write(f"**tan\u03b4** = {row['tan_delta']:.6f}  ({row[chr(215)+'quartz']:.2f}\u00d7 quartz)")
        st.write(f"**n_oxides** = {int(row['n_oxides'])}")
    with c2:
        st.write("**Composition (wt%)**")
        comp_df = pd.DataFrame(present.items(), columns=["Oxide", "wt%"]).set_index("Oxide")
        st.dataframe(comp_df.style.format({"wt%": "{:.1f}"}), width="stretch")
        st.bar_chart(comp_df)
