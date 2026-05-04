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
│   ├── baseline_lightgbm.py          # v1: Log + GroupKFold 기본
│   ├── baseline_lightgbm_v2.py       # v2: Layout 통합 + Rolling
│   ├── baseline_lightgbm_v3.py       # v3: 시계열 강화 (time_idx, cumulative_inflow)
│   ├── baseline_lightgbm_v4.py       # v4: 클러스터 TE + HRI/통신 지표
│   ├── baseline_catboost_v5.py       # v5: CatBoost 단일 모델 (일반화)
│   ├── baseline_catboost_v6.py       # v6: 피처 다이어트 (Permutation Importance)
│   ├── baseline_catboost_v7.py       # v7: PCA + Cyclic Time + Permutation
│   ├── baseline_catboost_v8.py       # v8: 3시드 앙상블 도입 (CatBoost)
│   ├── baseline_catboost_v9.py       # v9: CatBoost 3시드 앙상블 (블렌딩 제거)
│   ├── baseline_lightgbm_v10.py      # v10: LightGBM + v9 로직 (최신) ⭐
│   ├── eda_visualize.py              # EDA 이미지 생성
│   └── report_importance.py          # 중요도 시각화 분리
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

### v3 — 시계열 피처 강화
- **Temporal Sequencing**: `time_idx` (scenario 내 순번), `cumulative_inflow`
- **Smoothed Target Encoding**: Additive smoothing (alpha=10) 안정화
- **Extended Rolling Windows**: window=3,4,5로 확장
- **Log Transform Toggle**: `USE_LOG_TRANSFORM` 옵션으로 학습 방식 비교 가능

### v4 — 일반화 특화 (신규 레이아웃 대응)
- **Cluster-based Target Encoding**: layout_id → layout_cluster_id 기반 (K-Means k=10)
  - 처음 보는 layout_id에도 '유사 구조 창고' 평균 지연값 적용
- **Physical Bottleneck**: `path_complexity`, `congestion_persistence` 추가
- **Quantile Regression**: objective=quantile(alpha=0.55) + colsample_bytree=0.7
- **HRI + Infrastructure**: human_robot_density, network_bottleneck, environment_stress 상호작용

### v5 — CatBoost 전환 (일반화 중심)
- **Model Engine**: LightGBM → `CatBoostRegressor` 단일 모델
- **Loss Function**: Quantile(alpha=0.55) + MAE metric
- **Generalization**: layout_cluster_id 카테고리 변수로 명시적 선언
- **Feature Selection**: layout_id_target_enc, layout_cluster_target_enc 제거 (암기 신호 차단)
- **Ratio Features**: robot_active_per_area, inflow_per_aisle, congestion_vs_scenario_mean
- **Single Output**: 제출 CSV 단일화

### v6 — 피처 다이어트 + 상호작용
- **Permutation Importance**: 1차 학습 후 Top-150 피처만 선별 → 2차 재학습
- **Non-linear Interactions**: exp(congestion_score / 100) * sensor_noise_stress 등
- **Quantile Fine-tuning**: alpha=0.52로 조정
- **Log Transform Off**: 원본 MAE 직접 최적화

### v7 — PCA + Cyclic Time + Permutation
- **Layout PCA**: 물리 피처를 PCA(n_components=3) 압축 → 잠재 인수 추가
- **Cyclic Time Encoding**: 15분 슬롯을 sin/cos로 변환 (원형 성질 반영)
- **Extreme Feature Diet**: Permutation 기반 Top-110 피처만 선별
- **Deep Squeeze**: 모든 피처 공학 기법 적용 후 최소 피처셋으로 최고 성능 추구
- **Log Transform Restored**: 롱테일 분포 안정화 (USE_LOG_TRANSFORM=True)

### v8 — 3시드 앙상블 도입 (CatBoost)
- **Multi-Seed Ensemble**: seed=[42, 10, 2026] 각각 학습 후 평균 앙상블
- **Base**: v6 파이프라인 유지 (Top-150 피처, Quantile alpha=0.52)
- **Log Transform**: False 고정 (원본 MAE 직접 최적화)
- **Stage 구성**: Stage1 importance 평균 → Stage2 Top-150 재학습 → 3시드 평균

### v9 — CatBoost 3시드 앙상블 정제 (블렌딩 제거)
- **Blending 제거**: 단순 3시드 평균 앙상블만 유지 (불필요 복잡도 제거)
- **Base**: v8과 동일 구조 (Top-150, log_transform=False, Quantile)
- **Output**: `submission_v9_*.csv` 단일 출력

### v10 — LightGBM 복귀 + v9 로직 전체 이식 ⭐ (최신)
- **Model Engine**: CatBoost → `LGBMRegressor` 복귀 (objective=regression_l1)
- **Log Transform**: True 복원 (롱테일 안정화)
- **GroupKFold**: `layout_id` 기준으로 강화 (미지 레이아웃 일반화)
- **Physical Mechanics Features**: `saturation_index`, `explosion_factor`, `density_stress`, `workload_intensity`, `congestion_velocity` 추가
- **Layout PCA**: n_components=3 (layout_pca_capacity, complexity, efficiency)
- **Multi-Seed Ensemble**: seeds=[42, 10, 2026] 3시드 평균
- **Stage2 Top-150**: importance 평균 기반 피처 선별 재학습
- **총 피처 수**: ~352개

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

### 3. v1 학습
```bash
python src/baseline_lightgbm.py
# → data/submission/ 에 submission.csv 생성
```

### 4. v2 학습 (Layout 통합)
```bash
python src/baseline_lightgbm_v2.py
# → OOF MAE 출력
# → reports/eda/ 에 06~07 이미지 생성
# → data/submission/ 에 submission_v2_YYYYMMDD_HHMMSS.csv 생성
```

### 5. v3 학습 (시계열 강화)
```bash
python src/baseline_lightgbm_v3.py
```

### 6. v4 학습 (일반화 특화)
```bash
python src/baseline_lightgbm_v4.py
```

### 7. v5 학습 (CatBoost 단일 모델)
```bash
python src/baseline_catboost_v5.py
```

### 8. v6 학습 (피처 다이어트)
```bash
python src/baseline_catboost_v6.py
```

### 9. v7 학습 (PCA + Cyclic Time + Permutation)
```bash
python src/baseline_catboost_v7.py
```

### 10. v8 학습 (CatBoost 3시드 앙상블)
```bash
python src/baseline_catboost_v8.py
```

### 11. v9 학습 (CatBoost 3시드, 블렌딩 제거)
```bash
python src/baseline_catboost_v9.py
```

### 12. v10 학습 (최신: LightGBM + v9 로직) ⭐
```bash
conda activate dacon
python src/baseline_lightgbm_v10.py
# → data/submission/ 에 submission_v10_YYYYMMDD_HHMMSS.csv 생성
```

---

## 📈 Model Architecture

### LightGBM (v1~v4)
- **Framework**: LightGBM
- **Objective**: 
  - v1/v3: regression_l1 (MAE 직접 최적화)
  - v2/v4: quantile(alpha=0.55) (중잣값 기반 예측)
- **CV Strategy**: GroupKFold (N=5) by scenario_id
- **Hyperparameters** (v2):
  - learning_rate: 0.01
  - n_estimators: 3000
  - num_leaves: 127
  - subsample/colsample_bytree: 0.85

### CatBoost (v5~v9)
- **Framework**: CatBoost (Gradient Boosting on Decision Trees)
- **Objective**: 
  - v5: Quantile(alpha=0.55) + MAE
  - v6~v9: Quantile(alpha=0.52) + MAE
- **CV Strategy**: GroupKFold (N=5) by scenario_id
- **Key Parameters**:
  - loss_function: Quantile/MAE
  - eval_metric: MAE
  - early_stopping_rounds: 150
  - depth: 6~8
  - learning_rate: 0.1~0.05
  - cat_features: layout_cluster_id, layout_type_enc
- **v8/v9 Specifics**:
  - Multi-Seed Ensemble: seeds=[42, 10, 2026]
  - Stage1 importance 평균 → Stage2 Top-150 재학습
  - v9: 블렌딩 제거, 3시드 평균만 유지

### LightGBM v10 (최신) ⭐
- **Framework**: LightGBM (`LGBMRegressor`)
- **Objective**: `regression_l1` (MAE 직접 최적화)
- **CV Strategy**: GroupKFold (N=5) by `layout_id`
- **Key Parameters**:
  - learning_rate: 0.05
  - n_estimators: 3000
  - num_leaves: 127
  - subsample/colsample_bytree: 0.85/0.7
  - early_stopping_rounds: 150
- **v10 Specifics**:
  - Log Transform: True
  - Physical Mechanics Features: saturation_index, explosion_factor, density_stress, workload_intensity, congestion_velocity
  - Layout PCA: n_components=3
  - Multi-Seed Ensemble: seeds=[42, 10, 2026]
  - Stage2 Top-150 feature selection

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

- [ ] XGBoost 앙상블 (LightGBM 0.4 : CatBoost 0.3 : XGBoost 0.3)
- [ ] Optuna 하이퍼파라미터 최적화
- [ ] SHAP 값 기반 피처 해석
- [ ] 추가 시계열 피처 (ARIMA, LSTM)

---

## 📚 References

- **Project Notes**: [PROJECT_NOTES.md](PROJECT_NOTES.md)
- **EDA Results**: [reports/eda/](reports/eda/)
- **Python Version**: 3.13+
- **Key Libraries**: pandas, numpy, scikit-learn, lightgbm, catboost, matplotlib, seaborn

---

**Author**: dacon_user  
**Last Updated**: 2026-05-04  
**Latest Model**: v10 (LightGBM + v9 Logic + 3-Seed Ensemble) ⭐  
**Competition Status**: 참여 완료
