# Todo List

## Phase 1 — 라이브러리 레이어
- [ ] **T1** `recommend.py`: `_wt_to_mol_frame` → `wt_to_mol_frame` (public 노출, `__all__` 추가)

## Phase 2 — 데이터 레이어
- [ ] **T2** `app_glass.py` `run_search()`: mol% 변환 후 `batch_glass_probability` + `batch_thermal` 호출, 6개 컬럼 추가

## Phase 3 — UI 레이어
- [ ] **T3** `app_glass.py` 사이드바: P(glass) 슬라이더 추가 + `df_view` 필터 조건 추가
- [ ] **T4** `app_glass.py` 테이블: 6개 컬럼 + 단위 헤더 rename + format 추가
- [ ] **T5** `app_glass.py` detail view: Tg, Tx, Tliq, CTE, ΔT, P(glass) 표시 추가
