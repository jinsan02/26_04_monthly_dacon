# 스마트 창고 출고 지연 예측 | Smart Warehouse Outbound Delay Prediction

**Competition**: DACON 스마트 창고 출고 지연 예측  
**Date**: 2026-04-25  
**Status**: Private Repository  

---

## 📋 Project Overview

창고의 물리적 구조(레이아웃), 로봇 상태(배터리·가동), 주문 유입 등 다양한 변수를 종합 분석해 **출고 지연을 정확히 예측**하는 머신러닝 프로젝트입니다.

**Key Insight**: 배터리 부족 + 혼잡 동시 발생 상태가 지연의 최대 원인

---

## 🏗 Project Structure

```
dacon/
├── data/
│   ├── raw/                          # 학습·테스트 데이터
│   ├── meta/                         # 창고 레이아웃 메타정보
│   └── submission/                   # 제출 파일 저장
├── notebooks/                        # 원본 노트북 (참고용)
├── reports/
│   └── eda/                          # EDA 시각화 + 피처 중요도
├── src/
│   ├── baseline_lightgbm.py          # v1: 기본 개선사항 (로그변환, GroupKFold)
│   ├── baseline_lightgbm_v2.py       # v2: layout 통합 + 인사이트 반영 (최종)
│   └── eda_visualize.py              # EDA 이미지 생성
├── requirements.txt
└── PROJECT_NOTES.md                  # 상세 진행 기록
```

---

## 🔑 Key Improvements

### v1 — 기본 개선사항
- **Target Transform**: log1p 적용 (Long-tail 분포 정규화)
- **Validation Strategy**: GroupKFold (scenario_id 기준 데이터 누수 방지)
- **Missing Handling**: scenario 내 ffill/bfill → 중앙값 fallback
- **Interaction Features**: battery × congestion 복합 지표

### v2 — Layout 통합 + 고급 분석
- **Layout Integration**: 300개 창고 메타정보 Left Join
- **Physical Density Features**: 랙 밀집도, 교차로 복잡도, 통로 효율 등
- **Operational Efficiency**: charger_load, inflow_per_station 등
- **K-Means Clustering**: layout_group (비슷한 구조의 창고 그룹화)
- **Fold-safe Target Encoding**: layout_id 기준 과거 지연 통계
- **Rolling Window Features**: 3, 4 타임슬롯 기준 mean/max/std
- **Feature Importance-driven Pruning**: 중요도 낮은 is_missing 플래그 자동 제거

---

## 📊 Data Insights

| 항목 | 내용 |
|---|---|
| **Target Correlation Top 1** | `battery_mean` (배터리 평균) |
| **Critical State** | battery 부족 + congestion 높음 동시 발생 |
| **Layout Impact** | charger_ratio, aisle_compactness 상위 중요도 |
| **Temporal Pattern** | lag1(15분 전) 피처 상위권 (시계열 연속성) |

---

## 🚀 Quick Start

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. EDA 시각화
```bash
python src/eda_visualize.py
# → reports/eda/ 에 01~05 이미지 생성
```

### 3. v2 학습 (최종)
```bash
python src/baseline_lightgbm_v2.py
# → OOF MAE 출력
# → reports/eda/ 에 06~07 이미지 + CSV 생성
# → data/submission/ 에 submission_v2_YYYYMMDD_HHMMSS.csv 생성
```

---

## 📈 Model Architecture

- **Framework**: LightGBM
- **Objective**: regression_l1 (MAE 직접 최적화)
- **CV Strategy**: GroupKFold (N=5) by scenario_id
- **Hyperparameters**:
  - learning_rate: 0.01
  - n_estimators: 3000
  - num_leaves: 127
  - subsample/colsample_bytree: 0.85

---

## 💡 Advanced Features

### Physical/Structural
- `floor_area_per_robot`: 로봇 1대당 담당 면적
- `intersection_density`: 경로 복잡도
- `charger_ratio`: 충전소 부족도
- `aisle_compactness`: 통로 폭 대비 밀집도

### Operational (Runtime × Layout)
- `charger_load`: 실시간 충전소 부하
- `inflow_per_station`: 주문 처리 필요 인력
- `active_area_ratio`: 가동 로봇 대비 면적 여유

### State Combination
- `battery_stress`: battery / congestion (배터리-혼잡 트레이드오프)
- `battery_risk`: (1-battery) × low_battery_ratio (복합 위험도)

---

## 📝 Future Improvements

- [ ] CatBoost / XGBoost 앙상블 (가중 평균 0.4 : 0.3 : 0.3)
- [ ] Optuna 하이퍼파라미터 최적화
- [ ] learning_rate 추가 조정 (0.01 → 0.005)
- [ ] SHAP 값 기반 피처 해석

---

## 📚 References

- **Project Notes**: [PROJECT_NOTES.md](PROJECT_NOTES.md)
- **EDA Results**: [reports/eda/](reports/eda/)
- **Python Version**: 3.13+
- **Key Libraries**: pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn

---

**Author**: dacon_user  
**Last Updated**: 2026-04-25
