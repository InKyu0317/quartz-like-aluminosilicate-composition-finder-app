---
title: Glass Pilot
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Glass Pilot

GlassNet(glasspy 0.6) + VITRIFY 모델 기반 유리 조성 탐색 Streamlit 앱.

목표 유전 특성(tan δ, ε_r)에 가장 가까운 알루미노실리케이트 유리 조성을 랜덤 샘플링 + 스코어링합니다.

## 주요 기능

### 조성 탐색
- **Stratified sparse 샘플링**: 산화물 수 k=3~max_k를 균등 할당량으로 샘플링 → 소수-산화물 조성도 동일한 기회로 탐색
- **Iterative 필터링**: 목표 결과 수가 찰 때까지 배치 반복 샘플링 → ε_r 범위가 좁아도 결과 수 보장, 예산(50× 한도) 초과 시 경고 표시
- **Adaptive n_samples**: ε_r 범위 폭 + max_oxide_count에 비례해 자동 결정. 폭 ≥ 4 → base 20,000; max_k < K일 때 `base × full_groups / active_groups` 로 스케일업해서 recommend()에 전달 → 실제 출력 ~20,000 유지. 최소 base 5,000 (benchmark: 2k→33% top-1 일치, 5k→100%)
- **n_oxides quota 정규화 (max_k 독립성)**: per-k 할당량을 `n // (K - min_k + 1)` (full pool 고정)으로 계산 → max_oxide_count가 바뀌어도 k=3..min(max_k) 구간 샘플이 동일 (같은 seed 기준). 이전에는 max_k로 나눠 그룹수가 달라져 RNG 진행 순서가 달라졌음
- **n_oxides 조기 필터**: [3, max_n_oxides] 범위 외 조성은 샘플링 루프 내부에서 즉시 제거

### 모델 예측
- **유전 특성**: GlassNet으로 ε_r, tan δ 동시 예측 (단일 호출)
- **유리화 확률**: VITRIFY 모델로 P(glass) ∈ [0, 1] 예측
- **열물성**: Tg, Tx, Tliq, CTE, ΔT = Tx − Tg (단위: °C)
- **BO 후 자동 예측**: Bayesian Optimization 완료 직후 P(glass) + 열물성을 결과에 자동 보강

### UI
- **quartz 유사도 스코어**: tan δ 기준 정렬
- **필터**: 알칼리 포함/제외, ε_r 범위, 최대 산화물 수, P(glass) 하한, SiO₂ 최소, Al₂O₃ 2위 조건
- **배경 그라디언트 테이블**: score / tan δ / P(glass) / ε_r / ×quartz / n_oxides
- **BO 진행 표시**: Run BO 버튼 실행 중 비활성화 + 버튼 바로 아래 progress bar + iteration 상태 텍스트 표시

### Bayesian Optimization 정밀 탐색
- 랜덤 검색 상위 N개를 Seed 삼아 GP-BO(Expected Improvement)로 새로운 조성 탐색
- 수렴 곡선, 개선 이력 테이블(새 best 발견 시점만), 산화물 범위 차트(seed 범위 + BO best 오버레이) 제공
- BO 결과 테이블에 P(glass), Tg, Tx, Tliq, CTE, ΔT 열 포함

### 배포 이슈 수정 (HF Spaces)
- **glasspy 모델 Dockerfile 빌드 타임 다운로드**: `RUN python -c "import glasspy"` 추가 → 런타임 Zenodo 호출(JSONDecodeError) 방지
- **Streamlit 버전 핀 고정**: `requirements.txt`에 `streamlit==1.57.0` — frontend JS bundle hash 불일치(500 에러) 방지
- **glasspy PerformanceWarning 억제**: `warnings.filterwarnings` 로 내부 DataFrame fragmentation 경고 숨김

### 샘플링 안정성 개선 (2026-05)
- **n_samples floor 상향**: 2,000 → 5,000 (benchmark 실측: 2k→33% top-1 일치율, 5k→100%)
- **max_k 독립 quota 정규화**: per-k 할당량 분모를 `max_k`에서 `K - min_k + 1` (full pool 고정)으로 변경
  - 효과: max_oxide_count 변경 시 k=3..9 구간 샘플이 동일하게 유지 (같은 seed)
  - 이전: max_k=9 vs 11에서 RNG 진행 순서 달라져 top-1 조성이 완전히 바뀜
- **Auto target 표시**: max_oxide_count 변경에 따라 사이드바 Auto target 숫자도 연동 갱신

### 안정성 / 버그 수정
- **테이블 떨림 해결**: `section[data-testid='stMain']`에 `overflow-y: scroll` 강제 적용 → 수직 스크롤바 토글로 인한 10px 폭 진동(jitter) 완전 제거
- **Vega-Lite 경고 제거**: `st.bar_chart` → matplotlib 수평 막대 차트로 교체, 빈 데이터 가드 추가
- **슬라이더 값/step 불일치 수정**: ε_r 기본 하한 3.77 → 3.80 (0.1 step grid 정렬)
- **사이드바 테마 색상 경고 제거**: `config.toml` 4개 색상 명시

## 배포

- **Hugging Face Spaces**: https://huggingface.co/spaces/ip278116/quartz-like-aluminosilicate-composition-finder
- **GitHub**: https://github.com/InKyu0317/quartz-like-aluminosilicate-composition-finder-app

## 로컬 실행

```bash
# 의존성 설치 (conda 권장)
conda create -n glassenv python=3.11
conda activate glassenv
pip install -e .

# 앱 실행
streamlit run app_glass.py
```

## 프로젝트 구조

```
app_glass.py                  # Streamlit 대시보드
roadlab_matnav_lib/
  oxides.py                   # 산화물 몰질량 · normalize()
  predict.py                  # GlassPredictor (GlassNet + VITRIFY 래퍼)
  recommend.py                # sparse 샘플링 · iterative 필터링 · 스코어링
  bayesian_opt.py             # GP-BO (Matern ARD + EI) 정밀 조성 탐색
tests/
  test_apply_threshold.py     # _apply_threshold 단위/통합 테스트
  test_sparse_sampler.py      # _sample_sparse_subsets 테스트
  test_sampling_strategy.py   # 샘플링 전략 증명 테스트
  test_recommend_coverage.py  # recommend() 저산화물 커버리지 테스트
  test_topk_stability.py      # max_k 독립성 및 top-1 안정성 증명 테스트
tasks/
  plan.md                     # 기능 계획 문서
  todo.md                     # 완료 항목 추적
  bench_n_samples.py          # n_samples별 top-1 안정성/속도 벤치마크
requirements.txt
Dockerfile
```

## 모델 출처

- **GlassNet** / **VITRIFY**: [glasspy 0.6](https://github.com/drcassar/glasspy) (ONNX)
- 초기 실행 시 모델 파일 자동 다운로드 (~수십 초)

## 테스트

```bash
pytest tests/ -v
```

## 의존성

- Python ≥ 3.11
- glasspy ≥ 0.6
- streamlit==1.57.0
- numpy, pandas, matplotlib

## 구조

```
glass_composition_app/
├── app_glass.py              # Streamlit 앱 진입점
├── roadlab_matnav_lib/       # 유리 예측 라이브러리
│   ├── __init__.py
│   ├── oxides.py             # 산화물 카탈로그
│   ├── predict.py            # GlassNet 래퍼
│   └── recommend.py          # stratified sparse 샘플링 · iterative 필터링
├── tests/                    # 87개 단위·통합·회귀 테스트
├── tasks/                    # 개발 계획 및 분석 스크립트
├── pyproject.toml
└── .streamlit/
    └── config.toml
```
