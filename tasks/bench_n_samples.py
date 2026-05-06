"""Benchmark: how does n_samples affect top-K stability and runtime?

Question: should we raise n_samples ceiling/floor in app_glass.py?

Measures:
  - Per n_samples ∈ {2k, 5k, 10k, 20k, 50k}:
    - Top-1 composition agreement across 5 different seeds (Jaccard-style)
    - Wall-clock seconds for one full sample+predict+filter run
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from roadlab_matnav_lib.predict import GlassPredictor
from roadlab_matnav_lib.recommend import recommend

OXIDES = [
    'SiO2', 'Al2O3', 'MgO', 'CaO', 'SrO', 'BaO',
    'ZnO', 'ZrO2', 'B2O3', 'La2O3', 'Y2O3',
]

def _topk_sig(df: pd.DataFrame, k: int = 5) -> set[tuple[str, int]]:
    """Top-k oxide signatures of the #1 row (oxide name + rounded wt%)."""
    if df.empty:
        return set()
    row = df.iloc[0]
    return {(ox, round(float(row[ox]))) for ox in OXIDES if row.get(ox, 0) > 0}

def main():
    print("Loading predictor...")
    pred = GlassPredictor()
    pred.warm_up()

    grid = [2_000, 5_000, 10_000, 20_000, 50_000]
    print(f"\n{'n_samples':>10} | {'time(s)':>8} | {'top1 agree':>10} | top1 oxides")
    print("-" * 90)

    for n in grid:
        sigs = []
        wall = 0.0
        for seed in range(3):
            t0 = time.perf_counter()
            df = recommend(
                pred, OXIDES,
                eps_r_range=(3.5, 4.5),     # ±0.5 around quartz 3.77
                n_samples=n,
                max_n_oxides=11,
                oxide_threshold=1.0,
                score_weights=(1.0, 0.0),    # ε-distance only (deterministic given samples)
                seed=seed,
            )
            wall += time.perf_counter() - t0
            sigs.append(_topk_sig(df))
        # pairwise agreement
        if all(sigs):
            agreements = []
            for i in range(len(sigs)):
                for j in range(i+1, len(sigs)):
                    inter = len(sigs[i] & sigs[j])
                    union = len(sigs[i] | sigs[j])
                    agreements.append(inter / union if union else 0.0)
            agree = float(np.mean(agreements))
        else:
            agree = float('nan')

        ox_str = ", ".join(f"{o}:{w}" for o, w in sorted(sigs[0], key=lambda x: -x[1])[:4])
        print(f"{n:>10,} | {wall/3:>8.2f} | {agree*100:>9.1f}% | {ox_str}")

if __name__ == "__main__":
    main()
