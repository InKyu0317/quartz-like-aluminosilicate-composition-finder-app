# Feature Plan: VITRIFY 필터 + 열물성 컬럼 추가

## 요구사항
1. 사이드바에 VITRIFY P(glass) 0.0–1.0 슬라이더 추가 → 후처리 필터
2. 결과 테이블에 컬럼 추가: P(glass), ΔT (K), Tx (K), Tg (K), Tliquidus (K), CTE (×10⁻⁶/K)

---

# Phase 4: Stratified Sparse Sampling + Iterative Filtering (완료)

## 배경 / 문제

- Full-simplex Dirichlet(11개)에서 n_oxides ≤ 5 조성 확률 ≈ 0% → 우수한 소수-산화물 조성 탐색 불가
- max_n_oxides 슬라이더가 post-filter 역할만 했고 실제 샘플링에 반영 안 됨
- ε_r 범위가 좁을수록 버려지는 샘플이 많아져 결과 수 감소

## 해결책

### T7: Stratified Sparse Sampling
- k = 3 .. max_k 각 그룹에 균등 할당량(quota) 배정
- 각 k그룹이 quota를 채울 때까지 오버샘플링 → threshold 탈락에 강건
- vectorized(`np.put_along_axis`) 구현

### T10: Iterative Batch Sampling
- `n_samples` = ε_r·n_oxides 필터 통과 후 **목표 결과 수**
- acceptance rate 실시간 추적 → 다음 batch_size 자동 조정
- 예산: `n_samples × max_attempts_factor (=50)` 총 예측 한도
- budget 소진 시 `st.warning()` 경고 표시

### T11: n_oxides 조기 필터
- `recommend()` 내부 mask에 `(n_ox >= 3) & (n_ox <= effective_max_k)` 추가
- 샘플링 루프 안에서 즉시 제거 → 목표 수 카운트에서도 제외

## 결과

| max_k | n=3~k-2 비율 | n=max_k 비율 | 소요(n=2000, 넓은 범위) |
|-------|-------------|-------------|------------------------|
| 5     | ~36%        | ~28%        | 0.6s                   |
| 9     | ~15%        | ~7%         | 0.6s                   |
| 11    | ~12%        | ~4%         | 0.6s                   |

---

## 의존성 그래프

```
recommend.py
  _wt_to_mol_frame()      ← private (T1에서 public 노출 필요)
  recommend()             ← wt% DataFrame 반환

predict.py: GlassPredictor
  batch_glass_probability(mol_df)  ← VITRIFY 모델 (별도 ONNX)
  batch_thermal(mol_df)            ← GlassNet 1회 호출
    → Tg (K), Tx (K), Tliquidus (K), CTE_per_K (1/K), delta_T (K)

app_glass.py
  run_search()    ← T2: 위 두 메서드 추가 호출
  sidebar         ← T3: P(glass) 슬라이더 추가
  df_view filter  ← T3: p_glass >= p_glass_min 조건 추가
  display_cols    ← T4: 6개 컬럼 추가 + 단위 헤더
  detail view     ← T5: 새 값 표시
```

**핵심 제약**: `recommend()` 출력은 wt%, `batch_*` 메서드는 mol% 필요.
→ `run_search()` 안에서 `wt_to_mol_frame(df[oxide_cols])` 변환 후 호출.
→ **kept rows만 변환** (50k → 필터 후 수백 행) → 추가 비용 미미.

---

## 수직 태스크 슬라이스

### Phase 1 — 라이브러리 레이어 (T1)

#### T1: `_wt_to_mol_frame` public 노출
- **파일**: `roadlab_matnav_lib/recommend.py`
- **변경**: `_wt_to_mol_frame` → `wt_to_mol_frame` (underscore 제거), `__all__` 추가
- **수락 기준**:
  - `from roadlab_matnav_lib.recommend import wt_to_mol_frame` 성공
  - recommend() 내부 호출 이름 업데이트 확인
- **검증**: `python -c "from roadlab_matnav_lib.recommend import wt_to_mol_frame; print('OK')"`

---

### ✅ Checkpoint 1
```
python -c "from roadlab_matnav_lib.recommend import wt_to_mol_frame; print('OK')"
```

---

### Phase 2 — 데이터 레이어 (T2)

#### T2: `run_search()`에서 P(glass) + 열물성 컬럼 계산
- **파일**: `app_glass.py`
- **변경 위치**: `run_search()` 함수 내부, `recommend()` 호출 후
- **세부 단계**:
  1. `oxide_cols` 추출 (기존 코드 이동 불필요)
  2. `mol_df = wt_to_mol_frame(df[oxide_cols])` 로 mol% 변환
  3. `p_glass = predictor.batch_glass_probability(mol_df)` 호출
  4. `thermal = predictor.batch_thermal(mol_df)` 호출
  5. `df["p_glass"] = p_glass`
  6. `df["Tg_K"] = thermal["Tg"]`
  7. `df["Tx_K"] = thermal["Tx"]`
  8. `df["Tliq_K"] = thermal["Tliquidus"]`
  9. `df["CTE_1e6"] = thermal["CTE_per_K"] * 1e6`  — 소수라임 ≈ 9 기준
  10. `df["dT_K"] = thermal["delta_T"]`
- **수락 기준**:
  - `df.columns` 에 6개 신규 컬럼 존재
  - NaN 미전파 (GlassNet 예측 불가 조성은 NaN 허용, `.fillna(0.0)` 제거 필요)
  - `run_search` 캐시 시그니처 변경 없음 (파라미터 동일)
- **검증**:
  ```python
  # python -c 스크립트로 batch_thermal 출력 shape 확인
  ```

---

### ✅ Checkpoint 2
```
conda activate glassenv
python -c "
from roadlab_matnav_lib.predict import GlassPredictor
from roadlab_matnav_lib.recommend import wt_to_mol_frame
import pandas as pd

p = GlassPredictor()
test_wt = pd.DataFrame([{'SiO2':65,'Al2O3':15,'CaO':10,'MgO':10}])
mol = wt_to_mol_frame(test_wt)
print('P(glass):', p.batch_glass_probability(mol))
print(p.batch_thermal(mol))
"
```

---

### Phase 3 — UI 레이어 (T3, T4, T5)

#### T3: 사이드바 P(glass) 슬라이더 + 필터 적용
- **파일**: `app_glass.py`
- **변경**:
  1. 사이드바 `top_n` 슬라이더 아래에 추가:
     ```python
     p_glass_min = st.slider("P(glass) 하한", 0.0, 1.0, 0.5, 0.01)
     ```
  2. `df_view` 생성 라인 수정:
     ```python
     df_view = df[
         (df["n_oxides"] <= max_n_oxides) &
         (df["p_glass"] >= p_glass_min)
     ].head(top_n)
     ```
- **수락 기준**: 슬라이더 표시, 값 변경 시 테이블 즉시 갱신 (재검색 없음)

#### T4: 테이블 신규 컬럼 + 단위 헤더
- **파일**: `app_glass.py`
- **변경**:
  1. `display_cols` 에 추가 (eps_r 바로 앞 또는 score 뒤):
     ```python
     ["p_glass", "Tg_K", "Tx_K", "Tliq_K", "CTE_1e6", "dT_K"]
     ```
  2. `st.dataframe` 전에 rename으로 단위 표기:
     ```python
     df_view.rename(columns={
         "p_glass":  "P(glass)",
         "Tg_K":     "Tg (K)",
         "Tx_K":     "Tx (K)",
         "Tliq_K":   "Tliq (K)",
         "CTE_1e6":  "CTE (×10⁻⁶/K)",
         "dT_K":     "ΔT (K)",
     })
     ```
  3. 포맷 dict 추가:
     ```python
     "P(glass)":      "{:.2f}",
     "Tg (K)":        "{:.0f}",
     "Tx (K)":        "{:.0f}",
     "Tliq (K)":      "{:.0f}",
     "CTE (×10⁻⁶/K)": "{:.2f}",
     "ΔT (K)":        "{:.0f}",
     ```
- **수락 기준**: 컬럼 헤더에 단위 표시, NaN은 "-"로 표시

#### T5: Detail view 업데이트
- **파일**: `app_glass.py`, `Composition detail` 섹션
- **변경**: `c1` 컬럼에 신규 값 추가 표시
  ```python
  st.write(f"**P(glass)** = {row['p_glass']:.2f}")
  st.write(f"**Tg** = {row['Tg_K']:.0f} K")
  st.write(f"**Tx** = {row['Tx_K']:.0f} K  (ΔT = {row['dT_K']:.0f} K)")
  st.write(f"**Tliq** = {row['Tliq_K']:.0f} K")
  st.write(f"**CTE** = {row['CTE_1e6']:.2f} ×10⁻⁶/K")
  ```
- **수락 기준**: detail 패널에서 선택 행의 열물성 확인 가능

---

### ✅ Checkpoint 3 (최종)
```
conda activate glassenv
streamlit run app_glass.py
```
- 사이드바에 P(glass) 슬라이더 확인
- Run Search → 테이블에 6개 신규 컬럼 확인
- 슬라이더 조절 → 재검색 없이 테이블 즉시 필터링 확인
- Rank 입력 → detail에서 Tg, Tx, CTE 등 표시 확인

---

## 리스크 및 주의사항

| 리스크 | 대응 |
|--------|------|
| `batch_thermal`이 일부 조성에서 NaN 반환 | `.fillna(0.0)` 을 run_search 에서 제거하고 테이블 포맷에서 NaN→"-" 처리 |
| VITRIFY 모델 첫 로드 지연 | `load_predictor()` 가 `GlassPredictor` 를 캐시하므로 첫 Run Search 시 1회만 발생 |
| `recommend()` 반환 df에 전체 oxide_cols 없는 경우 | `wt_to_mol_frame(df[oxide_cols])` 에서 KeyError 가능 → `[c for c in oxide_cols if c in df.columns]` 사용 |
| CTE 단위 변환 오류 | glasspy memo: `CTEbelowTg` = log10(CTE in 1/K) → `batch_thermal`이 이미 `10**x` 디코딩 완료 → ×1e6만 필요 |
