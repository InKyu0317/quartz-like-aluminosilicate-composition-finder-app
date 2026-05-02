---
title: Quartz-like Aluminosilicate Composition Finder
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Glass Composition Finder

GlassNet 기반 유리 조성 탐색 Streamlit 앱.

## 기능

- 알칼리 프리 / 알칼리 포함 모드 전환
- ε_r 범위, 산화물 수, 샘플 수 조정
- quartz tan δ(0.000198) 기준 유사도 스코어링
- 조성 테이블 + 막대 그래프

## 실행

```bash
# 1. 의존성 설치
pip install -e .

# 2. 앱 실행
streamlit run app_glass.py
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
