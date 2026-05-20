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

## 배포

- **Hugging Face Spaces**: https://huggingface.co/spaces/ip278116/roadlab-glass-pilot
- **GitHub**: https://github.com/InKyu0317/roadlab-glass-pilot

## 주요 기능

### 사이드바 파라미터

| 위젯 | 설명 |
|------|------|
| Glass type | 유리 계열 선택 (선택 전 검색 비활성화) |
| Alkali content | Alkali-free / Include alkali 전환 |
| ε_r range | 유전율 범위 필터 (재검색 필요) |
| Target tanδ | 목표 유전 손실값 (기본: 0.000198, 석영 수준) |
| P(glass) 최소 | 유리화 확률 하한 |
| Tg 범위 (°C) | 유리전이온도 필터 |
| CTE 범위 (×10⁻⁶/K) | 열팽창계수 필터 |
| SiO₂ 최소 (wt%) | SiO₂ 함량 하한 |
| Max oxide count | 최대 산화물 종류 수 (재검색 필요) |
| Rows to display | 결과 테이블 표시 행 수 |
| Al₂O₃ 2위 조건 | Al₂O₃이 SiO₂ 다음으로 많은 조건 (알루미노실리케이트 정의) |

### 조성 탐색

- **Stratified sparse 샘플링**: 산화물 수 k=3~max_k를 균등 할당량으로 샘플링 → 소수-산화물 조성도 동일한 기회로 탐색
- **Iterative 필터링**: 목표 결과 수가 찰 때까지 배치 반복 샘플링 → ε_r 범위가 좁아도 결과 수 보장
- **Adaptive n_samples**: ε_r 범위 폭 + max_oxide_count에 비례해 자동 결정 (base 5,000~20,000)
- **tan δ 타겟 스코어링**: score = 1 / (1 + |tan δ - target| / target) — target에 가까울수록 높은 점수

### 모델 예측

- **유전 특성**: GlassNet으로 ε_r, tan δ 동시 예측
- **유리화 확률**: VITRIFY 모델로 P(glass) ∈ [0, 1] 예측
- **열물성**: Tg, Tx, Tliquidus, CTE, ΔT = Tx - Tg (단위: °C / ×10⁻⁶/K)
- 상위 2,000개 후보에만 VITRIFY + 열물성 예측 수행 (성능 최적화)

### Bayesian Optimization 정밀 탐색

- 랜덤 검색 상위 N개를 Seed 삼아 GP-BO (Expected Improvement)로 추가 탐색
- 수렴 곡선, 개선 이력 테이블, 산화물 범위 차트 제공
- BO 결과 테이블에 P(glass), Tg, Tx, Tliq, CTE, ΔT 포함

### UI

- 배경 그라디언트 테이블: score / tan δ / P(glass) / ε_r / ×target / n_oxides
- BO 진행 표시: Run BO 실행 중 버튼 비활성화 + progress bar + 반복 상태 텍스트
- 테이블 떨림 방지: overflow-y: scroll 강제 적용

## 로컬 실행

```bash
conda create -n glassenv python=3.11
conda activate glassenv
pip install -e .
streamlit run app_glass.py
```

## 프로젝트 구조

```
app_glass.py                      # Streamlit 대시보드 (메인 진입점)
roadlab_matnav_lib/
  __init__.py
  oxides.py                       # 산화물 몰질량 · normalize()
  predict.py                      # GlassPredictor (GlassNet + VITRIFY 래퍼)
  recommend.py                    # sparse 샘플링 · iterative 필터링 · 스코어링
  bayesian_opt.py                 # GP-BO (Matern ARD + EI) 정밀 탐색
tests/
  test_apply_threshold.py         # _apply_threshold 단위 테스트
  test_sparse_sampler.py          # 샘플러 단위 테스트
  test_sampling_strategy.py       # 샘플링 전략 증명 테스트
  test_recommend_coverage.py      # 저산화물 커버리지 테스트
  test_topk_stability.py          # max_k 독립성 · top-1 안정성 테스트
  test_recommend_score.py         # tan δ 스코어 방향 회귀 테스트
  test_ui_glass_type.py           # Glass type selectbox · 버튼 게이팅 테스트
  test_ui_tan_target.py           # target tanδ 슬라이더 · 시그니처 테스트
  test_ui_tg_cte.py               # Tg / CTE 슬라이더 · 필터 마스크 테스트
  test_ui_sidebar_order.py        # 사이드바 위젯 순서 테스트
Dockerfile
requirements.txt
pyproject.toml
```

## 테스트

```bash
pytest tests/ -v   # 120 tests
```

## 모델 출처

- **GlassNet** / **VITRIFY**: [glasspy 0.6](https://github.com/drcassar/glasspy) (ONNX)
- 초기 실행 시 모델 파일 자동 다운로드 (~수십 초)

## 의존성

- Python >= 3.11
- glasspy >= 0.6
- streamlit == 1.57.0
- numpy, pandas, matplotlib, scikit-learn