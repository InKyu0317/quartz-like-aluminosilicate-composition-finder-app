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
- [x] **Adaptive n_samples**: ε_r 범위 폭 기반 자동 결정 (폭 ≥ 4 → 20,000, 최소 2,000), 슬라이더 제거
- [x] **온도 단위**: Tg / Tx / Tliq / ΔT / CTE 단위 K → °C 전환 (표 + 상세 뷰)
- [x] **Bayesian Optimization**: `roadlab_matnav_lib/bayesian_opt.py` — GP(Matern ARD) + EI 순차 최적화
  - Seed: 랜덤 검색 상위 N개를 초기 관측값으로 사용
  - 매 iteration: GP 적합 → 후보 생성(Exploitation 75% + Exploration 25%) → EI 최대 지점 GlassNet 평가
  - 결과: 수렴 곡선, 개선 이력 테이블(새 best 발견 시점만), 산화물 범위 차트(seed 범위 + BO best 오버레이)

## 버그 수정 및 UI 개선 (2026-05)
- [x] **테이블 jitter 해결**: HF Spaces 좁은 viewport에서 테이블/메트릭이 10px 주기로 무한 진동하던 현상 수정
  - 원인: `section[data-testid='stMain']`의 `overflow-y: auto` → 수직 스크롤바 토글 피드백 루프
  - 수정: CSS `overflow-y: scroll !important` 강제 적용 (`st.markdown` unsafe_allow_html)
  - Playwright로 좁은 viewport(700px) 재현 후 80번 측정으로 unique=1 (jitter 완전 제거) 확인
- [x] **BO 진행 표시 개선**: Run BO 버튼 실행 중 비활성화 + 버튼 바로 아래 progress bar + iteration 상태 텍스트
  - 이전: `st.spinner` (전체 화면 차단) + 스크롤 아래 progress bar
  - 현재: 버튼 비활성화(`disabled`) + `st.empty()` placeholder로 버튼 옆에 즉시 표시
- [x] **BO 결과 열물성 보강**: BO 완료 후 P(glass) + Tg/Tx/Tliq/CTE/ΔT 자동 계산하여 BO 결과 테이블에 포함
- [x] **Vega-Lite 경고 제거**: 조성 상세 뷰 `st.bar_chart` → matplotlib `barh`로 교체, 빈 데이터 가드 추가
- [x] **슬라이더 grid 불일치 수정**: ε_r 슬라이더 기본값 3.77 → 3.80 (step=0.1 grid 정렬)
- [x] **사이드바 테마 색상 경고 제거**: `.streamlit/config.toml`에 4개 색상 명시
- [x] **불필요한 `width="small"` 제거**: 잘못된 jitter 진단으로 추가했던 column_config `width="small"` 전부 되돌림

## 배포 이슈 수정 및 샘플링 안정성 개선 (2026-05 후반)
- [x] **Streamlit 버전 핀 고정**: `requirements.txt`에 `streamlit==1.57.0` 명시 — HF Spaces에서 다른 버전 설치 시 frontend JS bundle hash 불일치로 `rehype-raw.*.js` 500 에러 + `<style>` 태그 텍스트 노출 발생
- [x] **glasspy 모델 빌드 타임 다운로드**: `Dockerfile`에 `RUN python -c "import glasspy"` 추가 — 런타임 Zenodo API 호출 실패(`JSONDecodeError`) 방지
- [x] **glasspy PerformanceWarning 억제**: `warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)` — 내부 DataFrame fragmentation 경고 로그 제거
- [x] **n_samples floor 상향 2,000 → 5,000**: benchmark 실측 근거 (2k→33% / 5k→100% top-1 일치율). `tasks/bench_n_samples.py` 벤치마크 스크립트 추가
- [x] **max_k 독립 quota 정규화**: `_sample_sparse_subsets`의 per-k 할당량 분모를 `max_k - min_k + 1` → `K - min_k + 1` (full pool 고정)으로 변경
  - 효과: max_oxide_count 변경 시 k=3..min(max_k) 구간 샘플 동일 유지 (같은 seed)
  - 이전 문제: max_k=9 vs 11에서 그룹수 달라 → RNG 진행 순서 다름 → top-1 조성 완전히 바뀜
  - Prove-It 테스트 추가: `tests/test_topk_stability.py`
- [x] **n_samples_auto max_k 연동**: `n_samples_auto = n_base × full_groups / active_groups` 로 스케일업 → recommend() 실제 출력 항상 ~n_base 유지. 사이드바 Auto target 표시도 실제 출력 수 기준으로 갱신
