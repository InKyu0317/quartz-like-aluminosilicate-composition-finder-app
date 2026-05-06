"""Streamlit dashboard — alkali-free aluminosilicate composition finder.

Run:
    streamlit run app_glass.py
"""
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from roadlab_matnav_lib.predict import GlassPredictor
from roadlab_matnav_lib.recommend import wt_to_mol_frame
import roadlab_matnav_lib as rml

# ── constants ────────────────────────────────────────────────────────────────
TAN_QUARTZ      = 0.000198
OXIDE_THRESHOLD = 1.0
FIXED           = {'SiO2': (40.0, 85.0)}
COL_XQUARTZ     = "×quartz"   # column key — defined once to avoid backslash in f-strings
VITRIFY_TOP_K   = 2_000      # limit VITRIFY/thermal to top-scored rows (perf)

ALKALI_FREE_OXIDES = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'ZrO2',
    'B2O3', 'La2O3', 'Y2O3',
]
WITH_ALKALI_OXIDES = ALKALI_FREE_OXIDES + ['Na2O', 'K2O', 'Li2O']

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quartz-like Aluminosilicate Composition Finder", layout="wide")
# Force vertical scrollbar always-visible on the main Streamlit section to prevent
# layout-jitter feedback loop. The culprit is section[data-testid="stMain"] which
# has overflow-y:auto — when content height oscillates around the visible area height,
# its scrollbar toggles on/off, stealing/returning ~10px width and causing all tables
# and metric blocks to shake. Forcing overflow-y:scroll pins the scrollbar permanently.
st.markdown(
    "<style>"
    "section[data-testid='stMain'] { overflow-y: scroll !important; }"
    "</style>",
    unsafe_allow_html=True,
)
st.title("Quartz-like Aluminosilicate Composition Finder")
st.caption(f"GlassNet · quartz reference: tan\u03b4 = {TAN_QUARTZ:.6f}, \u03b5_r = 3.77")

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Parameters")

    alkali_mode = st.radio("Alkali content", ["Alkali-free", "Include alkali"])
    active_oxides = ALKALI_FREE_OXIDES if alkali_mode == "Alkali-free" else WITH_ALKALI_OXIDES

    eps_min, eps_max = st.slider(
        "\u03b5_r range", min_value=3.0, max_value=15.0, value=(3.8, 10.0), step=0.1
    )
    max_n_oxides = st.slider(
        "Max oxide count", min_value=3, max_value=len(active_oxides), value=len(active_oxides),
        help="이 값을 변경하면 **Re-Run Search**가 필요합니다. 줄이면 해당 수 이하의 산화물 조합에서 직접 샘플링합니다."
    )
    top_n = st.slider("Rows to display", min_value=10, max_value=200, value=30)
    p_glass_min = st.slider(
        "P(glass) 최소", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="VITRIFY 모델 기준 유리화 확률 하한. 재검색 없이 즉시 적용됩니다."
    )
    sio2_min = st.slider(
        "SiO₂ 최소 (wt%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
        help="SiO₂ 함량 하한. 재검색 없이 즉시 적용됩니다."
    )
    al2o3_second = st.checkbox(
        "Al₂O₃ 2위 조건",
        value=True,
        help="Al₂O₃이 SiO₂ 다음으로 가장 많은 산화물이어야 함 (알루미노실리케이트 정의). 재검색 없이 즉시 적용됩니다."
    )
    run = st.button("Run Search", type="primary")

    st.divider()
    _eps_width = eps_max - eps_min
    _n_auto = int(np.clip(round(_eps_width / 4.0 * 20_000 / 1_000) * 1_000, 2_000, 20_000))
    st.caption(f"**Auto target**: {_n_auto:,}개 (ε_r 범위 {_eps_width:.1f} 기준)")
    st.divider()
    st.caption(f"**Oxide pool** ({len(active_oxides)} oxides)")
    st.caption("  ".join(f"`{ox}`" for ox in active_oxides))

# ── model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="GlassNet / VITRIFY 모델 로딩 중…")
def load_predictor():
    predictor = GlassPredictor()
    predictor.warm_up()   # load both ONNX models now (once), not on first search
    return predictor

# ── search (cached by params) ─────────────────────────────────────────────────
@st.cache_data(show_spinner="\uc870\uc131 \uc0d8\ud50c\ub9c1 \uc911\u2026", max_entries=8)
def run_search(oxide_tuple, eps_min, eps_max, n_samples, max_n_oxides, seed=0):
    predictor = load_predictor()
    oxides = list(oxide_tuple)
    use_sparse = max_n_oxides < len(oxides)
    df = rml.recommend.recommend(
        predictor,
        active_oxides=oxides,
        eps_r_range=(eps_min, eps_max),
        tan_delta_range=(0.0, 1.0),
        n_samples=n_samples,
        seed=seed,
        oxide_threshold=OXIDE_THRESHOLD,
        max_n_oxides=max_n_oxides,
    )

    oxide_cols = [c for c in oxides if c in df.columns]
    df["score"]    = 1.0 / (1.0 + np.abs(df["tan_delta"] - TAN_QUARTZ) / TAN_QUARTZ)
    df["n_oxides"] = (df[oxide_cols] > OXIDE_THRESHOLD).sum(axis=1).astype(int)
    df["\u00d7quartz"] = (df["tan_delta"] / TAN_QUARTZ).round(2)

    # Sort first so VITRIFY/thermal run only on the best-scoring rows
    df = df.sort_values(["score", "n_oxides"], ascending=[False, True]).reset_index(drop=True)
    n_total = len(df)   # candidates before VITRIFY trim (for metrics display)
    df = df.head(VITRIFY_TOP_K)

    # ── VITRIFY + thermal properties (top rows only) ─────────────────────────────
    mol_df = wt_to_mol_frame(df[oxide_cols].fillna(0.0))
    df["p_glass"] = predictor.batch_glass_probability(mol_df)
    thermal = predictor.batch_thermal(mol_df)
    df["Tg_K"]    = thermal["Tg"].to_numpy() - 273.15
    df["Tx_K"]    = thermal["Tx"].to_numpy() - 273.15
    df["Tliq_K"]  = thermal["Tliquidus"].to_numpy() - 273.15
    df["CTE_1e6"] = thermal["CTE_per_K"].to_numpy() * 1e6
    df["dT_K"]    = thermal["delta_T"].to_numpy()

    df.index += 1  # 1-based rank
    return df, oxide_cols, n_total, n_samples

# ── run only when button clicked ──────────────────────────────────────────────
# Auto-compute n_samples from ε_r range width.
# Anchor: width=4 → 20,000 (so any width≥4 hits the ceiling).
# Narrower ranges scale down proportionally. Floor: 2,000. Ceiling: 20,000.
_eps_width = eps_max - eps_min
n_samples_auto = int(np.clip(round(_eps_width / 4.0 * 20_000 / 1_000) * 1_000, 2_000, 20_000))

if run:
    df, oxide_cols, n_total, n_target = run_search(tuple(active_oxides), eps_min, eps_max, n_samples_auto, max_n_oxides)
    st.session_state["df"] = df
    st.session_state["searched_max_n_oxides"] = max_n_oxides
    st.session_state["oxide_cols"] = oxide_cols
    st.session_state["n_total"] = n_total
    st.session_state["n_target"] = n_target
    if n_total < n_target:
        st.warning(
            f"ε_r 범위가 좁아 목표 {n_target:,}개 중 **{n_total:,}개**만 수집됐습니다. "
            f"ε_r 범위를 넓히면 더 많은 후보를 얻을 수 있습니다."
        )

if "df" not in st.session_state:
    st.info("\uc67c\ucabd \uc0ac\uc774\ub4dc\ubc14\uc5d0\uc11c \ud30c\ub77c\ubbf8\ud130\ub97c \uc124\uc815\ud558\uace0 **Run Search** \ub97c \ub204\ub974\uc138\uc694.")
    st.stop()

df = st.session_state["df"]
oxide_cols = st.session_state["oxide_cols"]

# ── apply post-search filters (no rerun needed) ──────────────────────────────
sio2_mask = (df["SiO2"] >= sio2_min) if "SiO2" in df.columns else True
if al2o3_second and "Al2O3" in df.columns:
    other_oxide_cols = [c for c in oxide_cols if c not in ("SiO2", "Al2O3")]
    if other_oxide_cols:
        al2o3_mask = df[other_oxide_cols].lt(df["Al2O3"], axis=0).all(axis=1)
    else:
        al2o3_mask = True  # no other oxides to compare
else:
    al2o3_mask = True
# n_oxides range is guaranteed by recommend() — no additional filter needed here
df_view = df[
    (df["p_glass"] >= p_glass_min) &
    sio2_mask &
    al2o3_mask
].head(top_n)

# ── metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Target / Got", f"{st.session_state.get('n_target', '?'):,} / {st.session_state.get('n_total', len(df)):,}")
col2.metric("Shown (after filter)", f"{len(df_view):,}")
best = df_view.iloc[0] if len(df_view) else df.iloc[0]
col3.metric("Best ε_r", f"{best['eps_r']:.3f}")
col4.metric("Best tanδ / quartz", f"{best[COL_XQUARTZ]:.2f}×")
col5.metric("ε_r width → n", f"{eps_max-eps_min:.1f} → {st.session_state.get('n_target', n_samples_auto):,}")

# ── table ─────────────────────────────────────────────────────────────────────
COL_RENAME = {
    "p_glass":  "P(glass)",
    "Tg_K":     "Tg (°C)",
    "Tx_K":     "Tx (°C)",
    "Tliq_K":   "Tliq (°C)",
    "CTE_1e6":  "CTE (\u00d710\u207b\u2076/\u00b0C)",
    "dT_K":     "\u0394T (\u00b0C)",
}
display_cols = (
    ["eps_r", "tan_delta", "\u00d7quartz", "score", "n_oxides"]
    + ["p_glass", "Tg_K", "Tx_K", "Tliq_K", "CTE_1e6", "dT_K"]
    + [c for c in oxide_cols if c in df_view.columns]
)
df_display = df_view[display_cols].rename(columns=COL_RENAME)
_col_cfg = {
    "eps_r":                               st.column_config.NumberColumn("ε_r",  format="%.3f"),
    "tan_delta":                           st.column_config.NumberColumn("tanδ", format="%.6f"),
    "\u00d7quartz":                        st.column_config.NumberColumn(format="%.2f\u00d7"),
    "score":                               st.column_config.NumberColumn(format="%.4f"),
    "n_oxides":                            st.column_config.NumberColumn("#oxides"),
    "P(glass)":                            st.column_config.NumberColumn(format="%.2f"),
    "Tg (\u00b0C)":                        st.column_config.NumberColumn(format="%.0f"),
    "Tx (\u00b0C)":                        st.column_config.NumberColumn(format="%.0f"),
    "Tliq (\u00b0C)":                      st.column_config.NumberColumn(format="%.0f"),
    "CTE (\u00d710\u207b\u2076/\u00b0C)": st.column_config.NumberColumn(format="%.2f"),
    "\u0394T (\u00b0C)":                   st.column_config.NumberColumn(format="%.0f"),
    **{c: st.column_config.NumberColumn(format="%.1f") for c in oxide_cols},
}
_styled = (
    df_display.style
        .background_gradient(subset=["score"],         cmap="RdYlGn")
        .background_gradient(subset=["tan_delta"],     cmap="RdYlGn_r")
        .background_gradient(subset=["P(glass)"],      cmap="Blues")
        .background_gradient(subset=["eps_r"],         cmap="YlOrRd_r")
        .background_gradient(subset=["\u00d7quartz"],  cmap="RdYlGn_r")
        .background_gradient(subset=["n_oxides"],      cmap="Purples")
)
st.dataframe(
    _styled,
    width="stretch",
    height=600,
    column_config=_col_cfg,
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
        st.write(f"**tan\u03b4** = {row['tan_delta']:.6f}  ({row[COL_XQUARTZ]:.2f}\u00d7 quartz)")
        st.write(f"**n_oxides** = {int(row['n_oxides'])}")
        st.divider()
        st.write(f"**P(glass)** = {row['p_glass']:.2f}")
        st.write(f"**Tg** = {row['Tg_K']:.0f} °C")
        st.write(f"**Tx** = {row['Tx_K']:.0f} °C  (\u0394T = {row['dT_K']:.0f} °C)")
        st.write(f"**Tliq** = {row['Tliq_K']:.0f} °C")
        st.write(f"**CTE** = {row['CTE_1e6']:.2f} \u00d710\u207b\u2076/°C")
    with c2:
        st.write("**Composition (wt%)**")
        comp_df = pd.DataFrame(present.items(), columns=["Oxide", "wt%"]).set_index("Oxide")
        st.dataframe(comp_df, column_config={"wt%": st.column_config.NumberColumn(format="%.1f")}, width="stretch")
        if present:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _names = list(present.keys())
            _vals = list(present.values())
            _fig_c, _ax_c = plt.subplots(figsize=(4, max(len(_names) * 0.42 + 0.4, 2)))
            _ax_c.barh(_names, _vals, color="#AED6F1")
            _ax_c.set_xlabel("wt%")
            _ax_c.tick_params(axis="y", labelsize=9)
            _ax_c.invert_yaxis()
            _fig_c.tight_layout()
            st.pyplot(_fig_c)
            plt.close(_fig_c)

# ── Bayesian Optimization Refinement ─────────────────────────────────────────
st.divider()
with st.expander("🔬 Bayesian Optimization Refinement", expanded=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from roadlab_matnav_lib.bayesian_opt import run_bo as _run_bo

    st.caption(
        "검색 결과 상위 N개를 초기 관측값으로 삼아 GP-BO(Expected Improvement)로 tanδ를 최소화하는 조성을 탐색합니다. "
        "한 번에 1개 조성을 제안→평가→학습하는 순차적 최적화입니다."
    )

    bo_col1, bo_col2 = st.columns(2)
    with bo_col1:
        bo_seed_n = st.slider(
            "Seed compositions (top N)",
            min_value=5, max_value=min(50, max(len(df_view), 5)),
            value=min(20, max(len(df_view), 5)),
            key="bo_seed_n",
            help="현재 필터링된 결과 상위 N개를 GP 초기 학습 데이터로 사용합니다.",
        )
        bo_n_iter = st.slider(
            "BO iterations",
            min_value=5, max_value=100, value=30,
            key="bo_n_iter",
            help="각 iteration = GlassNet 1회 호출. 30회 기준 약 5~15초.",
        )
    with bo_col2:
        st.markdown("**현재 ε_r 범위**: `[{:.2f}, {:.2f}]` 로 고정 적용".format(eps_min, eps_max))
        st.markdown(
            "**Seed 최소 tanδ**: `{:.6f}` ({:.2f}×quartz)".format(
                df_view["tan_delta"].min() if len(df_view) else float("nan"),
                df_view["tan_delta"].min() / TAN_QUARTZ if len(df_view) else float("nan"),
            )
        )
        run_bo_btn = st.button("▶ Run BO Refinement", type="secondary", key="run_bo_btn",
                               disabled=st.session_state.get("bo_running", False))
        # Placeholders rendered immediately below the button — visible without scrolling
        _bo_status_ph  = st.empty()
        _bo_bar_ph     = st.empty()

    # ── Seed composition range chart ─────────────────────────────────────────
    # Always shown; BO best overlaid once BO has been run.
    st.markdown("---")
    st.markdown(f"**Seed 조성 탐색 범위** (상위 {bo_seed_n}개, 활성 산화물만 표시)")
    st.caption("bar = min–max 범위  |  ◆ = seed 평균  |  ● = seed best (최저 tanδ)  |  ★ = BO best (실행 후 표시)")

    _seed_data = df_view.head(bo_seed_n)[oxide_cols].fillna(0.0)
    _active = [c for c in oxide_cols if (_seed_data[c] > OXIDE_THRESHOLD).any()]
    if _active:
        _seed_sorted = df_view.head(bo_seed_n).sort_values("tan_delta")
        _seed_best_row = _seed_sorted.iloc[0]

        _bo_best_row = None
        if "bo_result" in st.session_state:
            _bo_pts = st.session_state["bo_result"]
            _bo_pts = _bo_pts[_bo_pts["source"] == "bo"]
            if len(_bo_pts):
                _bo_best_row = _bo_pts.sort_values("tan_delta").iloc[0]

        fig_r, ax_r = plt.subplots(figsize=(9, max(len(_active) * 0.48 + 0.5, 3)))
        for yi, ox in enumerate(_active):
            vals = _seed_data[ox]
            mn, mx, me = float(vals.min()), float(vals.max()), float(vals.mean())
            sb_val = float(_seed_best_row.get(ox, 0))
            # range bar
            ax_r.barh(yi, mx - mn, left=mn, height=0.5, color="#AED6F1", alpha=0.85,
                      label="seed range" if yi == 0 else "")
            # mean marker
            ax_r.plot(me, yi, "D", color="#2471A3", markersize=7, zorder=4,
                      label="seed mean" if yi == 0 else "")
            # seed best marker
            ax_r.plot(sb_val, yi, "o", color="#117A65", markersize=9, zorder=5,
                      label="seed best" if yi == 0 else "")
            # BO best marker (overlaid after run)
            if _bo_best_row is not None:
                bo_val = float(_bo_best_row.get(ox, 0))
                ax_r.plot(bo_val, yi, "*", color="#C0392B", markersize=15, zorder=6,
                          label="BO best" if yi == 0 else "")

        ax_r.set_yticks(range(len(_active)))
        ax_r.set_yticklabels(_active, fontsize=10)
        ax_r.set_xlabel("wt%")
        ax_r.set_xlim(left=-1)
        ax_r.set_title("Oxide composition space explored by seed")
        ax_r.legend(loc="lower right", fontsize=9)
        ax_r.grid(axis="x", alpha=0.3)
        fig_r.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

    # ── Run BO ───────────────────────────────────────────────────────────────
    if run_bo_btn:
        st.session_state["bo_running"] = True
        seed_df_bo = df_view.head(bo_seed_n)[oxide_cols + ["eps_r", "tan_delta"]].copy()
        _bo_status_ph.caption("⏳ GP-BO 준비 중…")
        progress_bar = _bo_bar_ph.progress(0.0)
        bo_log: list[float] = []

        def _bo_cb(i: int, total: int, best: float) -> None:
            bo_log.append(best)
            progress_bar.progress(i / total)
            _bo_status_ph.caption(
                f"🔄 Iteration {i}/{total} — best tanδ: {best:.6f} ({best/TAN_QUARTZ:.2f}×quartz)"
            )

        bo_result = _run_bo(
            load_predictor(),
            oxide_cols=oxide_cols,
            seed_df=seed_df_bo,
            n_iter=bo_n_iter,
            n_candidates=5_000,
            eps_r_range=(eps_min, eps_max),
            oxide_threshold=OXIDE_THRESHOLD,
            rng_seed=42,
            callback=_bo_cb,
        )

        progress_bar.progress(1.0)
        _bo_status_ph.caption(f"✅ 완료 — {bo_n_iter}회 반복")

        # ── Augment BO result with P(glass) + thermal properties ──────────────
        _bo_status_ph.caption("⏳ VITRIFY / thermal 예측 중…")
        _pred = load_predictor()
        _mol  = wt_to_mol_frame(bo_result[oxide_cols].fillna(0.0))
        bo_result["p_glass"] = _pred.batch_glass_probability(_mol)
        _th = _pred.batch_thermal(_mol)
        bo_result["Tg_K"]   = _th["Tg"].to_numpy() - 273.15
        bo_result["Tx_K"]   = _th["Tx"].to_numpy() - 273.15
        bo_result["Tliq_K"] = _th["Tliquidus"].to_numpy() - 273.15
        bo_result["CTE_1e6"] = _th["CTE_per_K"].to_numpy() * 1e6
        bo_result["dT_K"]   = _th["delta_T"].to_numpy()

        st.session_state["bo_result"]  = bo_result
        st.session_state["bo_log"]     = bo_log
        st.session_state["bo_running"] = False
        st.rerun()  # re-render to show updated range chart with BO best overlay

    # ── Persist BO results ────────────────────────────────────────────────────
    if "bo_result" in st.session_state:
        bo_result = st.session_state["bo_result"]
        bo_log    = st.session_state.get("bo_log", [])
        bo_only   = bo_result[bo_result["source"] == "bo"].copy()

        seed_best_s = float(df_view.head(bo_seed_n)["tan_delta"].min()) if len(df_view) else float("nan")
        bo_best_s   = float(bo_only["tan_delta"].min()) if len(bo_only) else seed_best_s
        improv_pct  = (seed_best_s - bo_best_s) / seed_best_s * 100.0 if seed_best_s > 0 else 0.0

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Seed best tanδ", f"{seed_best_s:.6f}", f"{seed_best_s/TAN_QUARTZ:.2f}×")
        m2.metric("BO best tanδ",   f"{bo_best_s:.6f}",   f"{bo_best_s/TAN_QUARTZ:.2f}×")
        m3.metric("Improvement", f"{improv_pct:+.1f}%",
                  delta_color="inverse" if improv_pct >= 0 else "normal")

        # ── Improvement trace ─────────────────────────────────────────────────
        if bo_log and "bo_iter" in bo_only.columns and len(bo_only):
            st.markdown("**개선 이력** — 새로운 best를 발견한 BO iteration만 표시")
            prev_best = float("inf")
            new_best_iters: list[int] = []
            for ii, v in enumerate(bo_log):
                if v < prev_best:
                    new_best_iters.append(ii + 1)
                    prev_best = v

            trace = bo_only[bo_only["bo_iter"].isin(new_best_iters)].copy()
            if len(trace):
                trace = trace.sort_values("bo_iter")
                trace[COL_XQUARTZ] = (trace["tan_delta"] / TAN_QUARTZ).round(2)
                trace["n_oxides"]  = (trace[oxide_cols].fillna(0) > OXIDE_THRESHOLD).sum(axis=1)
                _prop_cols = [c for c in ["p_glass", "Tg_K", "Tx_K", "Tliq_K", "CTE_1e6", "dT_K"] if c in trace.columns]
                _tc = (["bo_iter", "eps_r", "tan_delta", COL_XQUARTZ, "n_oxides"]
                        + _prop_cols
                        + [c for c in oxide_cols if c in trace.columns])
                _col_cfg_t = {
                    "eps_r":    st.column_config.NumberColumn("ε_r",   format="%.3f"),
                    "tan_delta": st.column_config.NumberColumn("tanδ",  format="%.6f"),
                    COL_XQUARTZ: st.column_config.NumberColumn(format="%.2f×"),
                    "bo_iter":  st.column_config.NumberColumn(),
                    "n_oxides": st.column_config.NumberColumn("#oxides"),
                    "p_glass":  st.column_config.NumberColumn("P(glass)", format="%.2f"),
                    "Tg_K":     st.column_config.NumberColumn("Tg (°C)",   format="%.0f"),
                    "Tx_K":     st.column_config.NumberColumn("Tx (°C)",   format="%.0f"),
                    "Tliq_K":   st.column_config.NumberColumn("Tliq (°C)", format="%.0f"),
                    "CTE_1e6":  st.column_config.NumberColumn("CTE (×10⁻⁶/°C)", format="%.2f"),
                    "dT_K":     st.column_config.NumberColumn("ΔT (°C)",   format="%.0f"),
                    **{c: st.column_config.NumberColumn(format="%.1f") for c in oxide_cols},
                }
                _styled_t = (
                    trace[_tc].style
                        .background_gradient(subset=["tan_delta"], cmap="RdYlGn_r")
                )
                if "p_glass" in trace.columns:
                    _styled_t = _styled_t.background_gradient(subset=["p_glass"], cmap="Blues")
                st.dataframe(_styled_t, column_config=_col_cfg_t, width="stretch")

        # ── Convergence chart ──────────────────────────────────────────────────
        if bo_log:
            cummin = np.minimum.accumulate(bo_log)
            # Mark iterations where best improved
            prev_b = float("inf")
            improv_iters, improv_vals = [], []
            for ii, v in enumerate(bo_log):
                if v < prev_b:
                    improv_iters.append(ii + 1)
                    improv_vals.append(v)
                    prev_b = v

            fig_c, ax_c = plt.subplots(figsize=(9, 3))
            ax_c.plot(range(1, len(bo_log) + 1), bo_log, "o",
                      color="#AAAAAA", markersize=4, label="iteration result")
            ax_c.plot(range(1, len(cummin) + 1), cummin, "b-",
                      linewidth=2, label="cumulative best")
            ax_c.plot(improv_iters, improv_vals, "*",
                      color="#C0392B", markersize=12, zorder=5, label="new best")
            ax_c.axhline(TAN_QUARTZ, color="gray", linestyle="--", linewidth=1,
                         label=f"quartz ({TAN_QUARTZ:.6f})")
            ax_c.set_xlabel("BO iteration")
            ax_c.set_ylabel("tanδ")
            ax_c.set_title("GP-BO Convergence  (★ = new best found)")
            ax_c.legend(fontsize=9)
            ax_c.grid(alpha=0.3)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        # ── Full BO compositions table ─────────────────────────────────────────
        if len(bo_only):
            bo_only[COL_XQUARTZ] = (bo_only["tan_delta"] / TAN_QUARTZ).round(2)
            bo_only["n_oxides"]  = (bo_only[oxide_cols].fillna(0) > OXIDE_THRESHOLD).sum(axis=1)
            st.markdown("**전체 BO 발견 조성** (tanδ 오름차순)")
            _prop_cols_b = [c for c in ["p_glass", "Tg_K", "Tx_K", "Tliq_K", "CTE_1e6", "dT_K"] if c in bo_only.columns]
            _bc = (["eps_r", "tan_delta", COL_XQUARTZ, "n_oxides"]
                    + (["bo_iter"] if "bo_iter" in bo_only.columns else [])
                    + _prop_cols_b
                    + [c for c in oxide_cols if c in bo_only.columns])
            _col_cfg_b = {
                "eps_r":    st.column_config.NumberColumn("ε_r",   format="%.3f"),
                "tan_delta": st.column_config.NumberColumn("tanδ",  format="%.6f"),
                COL_XQUARTZ: st.column_config.NumberColumn(format="%.2f×"),
                "bo_iter":  st.column_config.NumberColumn(),
                "n_oxides": st.column_config.NumberColumn("#oxides"),
                "p_glass":  st.column_config.NumberColumn("P(glass)", format="%.2f"),
                "Tg_K":     st.column_config.NumberColumn("Tg (°C)",   format="%.0f"),
                "Tx_K":     st.column_config.NumberColumn("Tx (°C)",   format="%.0f"),
                "Tliq_K":   st.column_config.NumberColumn("Tliq (°C)", format="%.0f"),
                "CTE_1e6":  st.column_config.NumberColumn("CTE (×10⁻⁶/°C)", format="%.2f"),
                "dT_K":     st.column_config.NumberColumn("ΔT (°C)",   format="%.0f"),
                **{c: st.column_config.NumberColumn(format="%.1f") for c in oxide_cols},
            }
            _bo_top = bo_only[_bc].sort_values("tan_delta").head(30)
            _styled_b = (
                _bo_top.style
                    .background_gradient(subset=["tan_delta"], cmap="RdYlGn_r")
                    .background_gradient(subset=["eps_r"],     cmap="YlOrRd_r")
            )
            if "p_glass" in _bo_top.columns:
                _styled_b = _styled_b.background_gradient(subset=["p_glass"], cmap="Blues")
            st.dataframe(
                _styled_b,
                column_config=_col_cfg_b,
                width="stretch",
                height=400,
            )
