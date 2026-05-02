# Todo List

## Phase 1 — 라이브러리 레이어
- [x] **T1** `recommend.py`: `_wt_to_mol_frame` → `wt_to_mol_frame` (public 노출, `__all__` 추가)

## Phase 2 — 데이터 레이어
- [x] **T2** `app_glass.py` `run_search()`: mol% 변환 후 `batch_glass_probability` + `batch_thermal` 호출, 6개 컬럼 추가

## Phase 3 — UI 레이어
- [x] **T3** `app_glass.py` 사이드바: P(glass) 슬라이더 추가 + `df_view` 필터 조건 추가
- [x] **T4** `app_glass.py` 테이블: 6개 컬럼 + 단위 헤더 rename + format 추가
- [x] **T5** `app_glass.py` detail view: Tg, Tx, Tliq, CTE, ΔT, P(glass) 표시 추가

## Phase 4 — 샘플링 개선
- [x] **T6** n_oxides 분포 분석 (`tasks/_analyze_n_oxides.py` 실행)
- [x] **T7** `recommend.py`: `_sample_sparse_subsets` 구현 — k별 stratified quota + vectorized `np.put_along_axis`
- [x] **T8** `recommend.py`: `recommend()` 항상 sparse 경로 사용 (max_n_oxides=11 시 full-simplex 경로 제거)
- [x] **T9** `app_glass.py`: `max_n_oxides` 사이드바 슬라이더 → search parameter로 전환 (min=3)
- [x] **T10** `recommend.py`: iterative 샘플링 — 목표 결과 수가 찰 때까지 배치 반복, adaptive 배치 크기
- [x] **T11** `recommend.py`: mask에 n_oxides 범위 조건 추가 → 조기 필터링
- [x] **T12** `app_glass.py`: Target candidates 슬라이더 (기본값 2,000), 목표 미달 시 경고 표시

## 추가 완료 항목
- [x] **성능**: `batch_eps_tan()` 단일 GlassNet 호출 / `VITRIFY_TOP_K=2000` 상위 행만 처리
- [x] **데이터 정확성**: `_apply_threshold()` — 1 wt% 미만 산화물 제거 후 재정규화 **모델 호출 전** 적용
- [x] **UI**: 앱 타이틀 변경 / 산화물 풀 표시 / 배경 그라디언트 / Al₂O₃ 2위 조건 / SiO₂ 최소 필터
- [x] **테스트**: 87개 단위·통합·회귀 테스트 (4개 파일)
- [x] **배포**: Hugging Face Spaces (Docker) 배포
