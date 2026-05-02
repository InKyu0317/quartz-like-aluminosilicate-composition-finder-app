---
title: Quartz-like Aluminosilicate Composition Finder
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Quartz-like Aluminosilicate Composition Finder

GlassNet(glasspy 0.6) + VITRIFY 모델 기반 알루미노실리케이트 유리 조성 탐색 Streamlit 앱.

quartz(석영) 수준의 유전 특성(tan δ ≈ 0.000198, ε_r ≈ 3.77)을 목표로 최적 조성을 랜덤 샘플링 + 스코어링합니다.

## 주요 기능

- **조성 샘플링**: Dirichlet 분포 기반 simplex 샘플링 (SiO2 범위 고정)
- **산화물 임계값**: 1 wt% 미만 산화물 제거 후 재정규화 → 모델 입력 정확성 보장
- **유전 특성 예측**: GlassNet으로 ε_r, tan δ 동시 예측 (단일 호출)
- **유리화 확률**: VITRIFY 모델로 P(glass) ∈ [0, 1] 예측
- **열물성 예측**: Tg, Tx, Tliq, CTE, ΔT = Tx − Tg
- **quartz 유사도 스코어**: tan δ 기준 정렬
- **UI 필터**: 알칼리 포함/제외, ε_r 범위, 산화물 수, P(glass) 하한
- **배경 그라디언트 테이블**: score / tan δ / P(glass) / ε_r / ×quartz / n_oxides

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
  recommend.py                # 심플렉스 샘플링 · 임계값 처리 · 스코어링
tests/
  test_apply_threshold.py     # _apply_threshold 단위/통합 테스트 (15개)
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
- streamlit ≥ 1.35
- numpy, pandas

## 구조

```
glass_composition_app/
├── app_glass.py              # Streamlit 앱 진입점
├── roadlab_matnav_lib/       # 유리 예측 라이브러리
│   ├── __init__.py
│   ├── oxides.py             # 산화물 카탈로그
│   ├── predict.py            # GlassNet 래퍼
│   └── recommend.py          # 조성 샘플링 및 추천
├── pyproject.toml
└── .streamlit/
    └── config.toml
```
