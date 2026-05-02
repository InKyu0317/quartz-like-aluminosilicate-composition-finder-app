# Todo List

## Phase 1 — 라이브러리 레이어
- [x] **T1** `recommend.py`: `_wt_to_mol_frame` → `wt_to_mol_frame` (public 노출, `__all__` 추가)

## Phase 2 — 데이터 레이어
- [x] **T2** `app_glass.py` `run_search()`: mol% 변환 후 `batch_glass_probability` + `batch_thermal` 호출, 6개 컬럼 추가

## Phase 3 — UI 레이어
- [x] **T3** `app_glass.py` 사이드바: P(glass) 슬라이더 추가 + `df_view` 필터 조건 추가
- [x] **T4** `app_glass.py` 테이블: 6개 컬럼 + 단위 헤더 rename + format 추가
- [x] **T5** `app_glass.py` detail view: Tg, Tx, Tliq, CTE, ΔT, P(glass) 표시 추가

## 추가 완료 항목
- [x] **성능**: `batch_eps_tan()` 단일 GlassNet 호출 / `VITRIFY_TOP_K=2000` 상위 행만 처리 → 145s → 11s
- [x] **데이터 정확성**: `_apply_threshold()` — 1 wt% 미만 산화물 제거 후 재정규화 **모델 호출 전** 적용
- [x] **UI**: 앱 타이틀 변경 / 산화물 풀 표시 / 배경 그라디언트 (score, tan δ, P(glass), ε_r, ×quartz, n_oxides)
- [x] **테스트**: `tests/test_apply_threshold.py` 15개 단위·통합·회귀 테스트 작성
- [x] **배포**: Hugging Face Spaces (Docker) 배포
