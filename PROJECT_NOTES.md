# 스마트 창고 출고 지연 예측 — 프로젝트 노트

## 1. 프로젝트 구조

```
dacon/
├── data/
│   ├── raw/
│   │   ├── train.csv              # 학습 데이터 (250,000 행 × 94 컬럼)
│   │   └── test.csv               # 테스트 데이터
│   ├── meta/
│   │   └── layout_info.csv        # 창고 레이아웃 메타 (300 행 × 15 컬럼)
│   └── submission/
│       ├── sample_submission.csv
│       ├── submission.csv          # v1 제출 결과
│       ├── submission_v2.csv       # v2 제출 결과 ✅
│       ├── submission_v3_*.csv     # v3 제출 결과 (timestamp)
│       ├── submission_v4_*.csv     # v4 제출 결과 (timestamp)
│       └── submission_v5_*.csv     # v5 단일 CatBoost 제출 결과
├── notebooks/
│   └── [Baseline]_LightGBM 기반 스마트 창고 출고 지연 예측.ipynb
├── reports/
│   └── eda/
│       ├── 01_missing_ratio_top30.png          # 결측치 비율 상위 30개
│       ├── 02_target_distribution.png          # 타깃 분포 (Long-tail)
│       ├── 03_target_correlation_top20.png     # 타깃 상관도 Top 20
│       ├── 04_correlation_heatmap_top35.png    # 상관관계 히트맵 Top 35
│       ├── 05_pairplot_top_features.png        # 주요 피처 Pairplot ✅
│       ├── 06_feature_importance_top40.png     # 전체 피처 중요도 Top 40 ✅
│       ├── 07_layout_feature_importance.png    # layout 파생 피처 중요도 ✅
│       ├── 08_feature_importance_v3_top40.png  # v3/v4 전체 피처 중요도 Top 40
│       ├── 09_layout_feature_importance_v3.png # v3/v4 layout 피처 중요도
│       ├── 10_v3_new_features_importance.png   # v3 신규 피처 중요도
│       ├── feature_importance.csv              # v2 중요도 수치 ✅
│       └── feature_importance_v3.csv           # v3/v4 중요도 수치 ✅
├── src/
│   ├── baseline_lightgbm.py        # v1 개선 베이스라인
│   ├── baseline_lightgbm_v2.py     # v2 최종 코드 (layout 통합)
│   ├── baseline_lightgbm_v3.py     # v3 개선 코드
│   ├── baseline_lightgbm_v4.py     # v4 일반화 특화 코드
│   ├── baseline_lightgbm_v5.py     # v5 CatBoost + 제출물 중심 코드 (최신)
│   ├── report_importance.py        # 중요도 시각화 분리 스크립트
│   └── eda_visualize.py            # EDA 시각화 스크립트
├── requirements.txt
└── PROJECT_NOTES.md
```

---

## 2. 진행 순서 요약

### Step 1 — 디렉토리 재구성
- 원본 단일 폴더(`open/`) → `data/raw`, `data/meta`, `data/submission` 분리
- 노트북 →`notebooks/` 이동
- 노트북 코드 셀 추출 → `src/baseline_lightgbm.py` 생성

### Step 2 — 의존성 정리
- `requirements.txt` 생성 및 패키지 설치

```
numpy
pandas
scikit-learn
lightgbm
matplotlib
seaborn
```

### Step 3 — 경로 오류 수정
- 데이터 경로를 `Path(__file__).resolve().parents[1]` 기준 절대경로로 통일
- `data/raw/train.csv`, `data/raw/test.csv`, `data/submission/submission.csv`

### Step 4 — EDA 시각화 (`src/eda_visualize.py`)
생성된 이미지 목록:

| 파일 | 내용 | 상태 |
|---|---|---|
| `01_missing_ratio_top30.png` | 결측치 비율 상위 30개 피처 | ✅ |
| `02_target_distribution.png` | 타깃 변수 분포 (Long-tail 확인) | ✅ |
| `03_target_correlation_top20.png` | 타깃과 상관도 상위 20개 피처 | ✅ |
| `04_correlation_heatmap_top35.png` | 타깃 기준 상위 35개 피처 상관관계 히트맵 | ✅ |
| `05_pairplot_top_features.png` | 타깃 상관 상위 피처 Pairplot (5,000샘플) | ✅ |

실행:
```bash
python src/eda_visualize.py
```

### Step 5 — v1 베이스라인 개선 (`src/baseline_lightgbm.py`)

| 항목 | 내용 |
|---|---|
| 타깃 변환 | `np.log1p()` 적용, 예측 시 `np.expm1()` 복원 + `clip(lower=0)` |
| 목적함수 | `regression_l1` (MAE 직접 최적화) |
| 검증 전략 | `KFold` → `GroupKFold(scenario_id)` (데이터 누수 방지) |
| 결측치 처리 | `scenario_id` 내 `ffill/bfill` → 중앙값 fallback |
| 결측치 플래그 | `{col}_is_missing` 피처 자동 생성 |
| 상호작용 피처 | `charging_per_active`, `low_battery_x_battery_mean`, `inflow_x_congestion` |
| Lag/Diff 피처 | `order_inflow_15m`, `congestion_score`, `low_battery_ratio` 기준 lag1/2, diff1 |
| 시나리오 통계 | `scenario_id` 기준 mean/std 피처 |

### Step 6 — v2 (`src/baseline_lightgbm_v2.py`)

v1 대비 추가 사항:

| 항목 | 내용 |
|---|---|
| Rolling 피처 | window=3,4 기준 mean/max/std (shift(1)로 누수 차단) |
| 하이퍼파라미터 | `learning_rate` 0.03 → 0.01, `n_estimators` 1500 → 3000 |
| Early Stopping | patience 100 → 150 |
| 모듈화 | `build_lgbm()`, `build_features()`, `run_cv()` 분리 → 앙상블 확장 구조 |

### Step 7 — layout_info.csv 통합 (v2에 반영)

**`layout_id` 기준 Left Join 후 다음 피처 생성:**

#### 물리적 밀집도 / 공간 복잡도 (정적 피처)

| 피처 | 계산 방법 | 의도 |
|---|---|---|
| `floor_area_per_robot` | `floor_area_sqm / (robot_total + 1)` | 로봇 1대당 담당 면적 |
| `intersection_density` | `intersection_count / (floor_area_sqm + 1)` | 경로 복잡도 |
| `aisle_compactness` | `layout_compactness / (aisle_width_avg + 1e-3)` | 통로 폭 대비 밀도 |
| `route_constraint` | `one_way_ratio × intersection_count` | 경로 제약 복합 지표 |
| `charger_ratio` | `robot_total / (charger_count + 1)` | 충전소 부족도 |
| `pack_station_per_area` | `pack_station_count / (floor_area_sqm + 1)` | 패킹 스테이션 밀도 |

#### 운영 효율 피처 (layout × train 결합)

| 피처 | 계산 방법 | 의도 |
|---|---|---|
| `charger_load` | `robot_active / (charger_count + 1)` | 실시간 충전소 부하 |
| `inflow_per_station` | `order_inflow_15m / (pack_station_count + 1)` | 통로 혼잡 예상치 |
| `active_area_ratio` | `floor_area_per_robot / (robot_active + 1)` | 가동 로봇 대비 면적 여유 |

#### 범주형 레이아웃 처리

| 피처 | 방법 |
|---|---|
| `layout_group` | K-Means(k=6) 클러스터링, `astype("category")` 선언 |
| `layout_id_target_enc` | GroupKFold 기반 fold-safe 타깃 인코딩 |
| `layout_type_enc` | label encoding + `astype("category")` 선언 |

### Step 8 — 피처 중요도 시각화 (v2에 반영)

v2 학습 완료 후 자동 저장 (✅ 생성 완료):

| 파일 | 내용 |
|---|---|
| `06_feature_importance_top40.png` | 전체 Top 40 중요도 막대 그래프 |
| `07_layout_feature_importance.png` | layout 파생 피처만 강조 |
| `feature_importance.csv` | 전체 피처 중요도 수치 |

### Step 9 — v3 개선 (OOF/Public 괴리 대응)

v2 대비 추가 사항:

| 항목 | 내용 |
|---|---|
| Smoothed Target Encoding 강화 | Additive smoothing(`alpha=10`) 유지 + unknown fallback 안정화 |
| 시계열 피처 추가 | `time_idx`(시나리오 내 순번), `cumulative_inflow`, `cumulative_inflow_lag1` |
| Rolling 확장 | window=3,4 → 3,4,5 |
| 로그 변환 실험 토글 | `USE_LOG_TRANSFORM` 상수로 log 학습/원본 학습 비교 가능 |
| 출력 분리 | 학습 스크립트는 submission + importance CSV만 저장 |

### Step 10 — v4 일반화 특화 개선 (신규 레이아웃 대응)

핵심 목표: **처음 보는 layout_id에서도 성능 유지**

| 전략 | 내용 |
|---|---|
| 전략 1: 클러스터 기반 TE | `layout_id_target_enc` → `layout_cluster_target_enc` 전환 |
|  | layout_info 물리 피처로 K-Means(`k=10`) 클러스터 생성 후 Fold-safe encoding |
| 전략 2: 물리 병목 강화 | `path_complexity = (floor_area_sqm / obstacle_ratio) / rack_count` 추가 |
|  | `congestion_persistence`(최근 3슬롯 congestion std) 추가 |
| 전략 3: 학습 조건 조정 | `objective="quantile"`, `alpha=0.55` 적용 |
|  | `colsample_bytree=0.7`로 특정 피처 의존도 완화 |

### Step 11 — 중요도 리포트 스크립트 분리

- `src/report_importance.py` 신설
- 입력: `reports/eda/feature_importance_v3.csv`
- 출력:
  - `08_feature_importance_v3_top40.png`
  - `09_layout_feature_importance_v3.png`
  - `10_v3_new_features_importance.png`

### Step 12 — v5 (Back to Basics + Smart Ratio + CatBoost)

핵심 목표: **단일 모델로 일반화 성능 극대화 (Unseen layout 대응)**

| 항목 | 내용 |
|---|---|
| 모델 엔진 | `CatBoostRegressor` 단일 모델 사용 |
| 손실함수 | `loss_function="Quantile:alpha=0.55"`, `eval_metric="MAE"` |
| 로그 변환 | `USE_LOG_TRANSFORM=True` 유지 (롱테일 안정화) |
| 레이아웃 일반화 | layout 물리 피처 기반 `layout_cluster_id`(k=15) 생성 후 `cat_features`로 학습 |
| ID/TE 제어 | `layout_id_target_enc`, `layout_cluster_target_enc` 제거 (암기 신호 차단) |
| Ratio 피처 | `robot_active_per_area`, `inflow_per_aisle`, `congestion_vs_scenario_mean` |
| 상호작용 피처 | `ops_pressure_idx`, `human_robot_density`, `congestion_tenure_weighted`, `network_instability`, `environment_stress` |
| 출력 정책 | 제출물 단일 CSV만 저장 (`submission_v5_*.csv`) |

v5 생성 결과:
- `submission_v5_*.csv`

### Step 13 — v6 (피처 다이어트 + 상호작용 강화)

핵심 목표: **Permutation Importance 기반 피처 선별 → 과적합 최소화**

| 항목 | 내용 |
|---|---|
| 모델 엔진 | `CatBoostRegressor` 단일 모델 |
| 손실함수 | `loss_function="Quantile:alpha=0.52"`, `eval_metric="MAE"` (v5에서 0.55→0.52로 조정) |
| 로그 변환 | `USE_LOG_TRANSFORM=False` (원본 MAE 직접 최적화) |
| 피처 선별 전략 | 1차 학습 후 Permutation Importance 기반 Top-150 피처 선택 → 2차 재학습 |
| 레이아웃 클러스터링 | K-Means(k=10) 유지 |
| 비선형 상호작용 | `exp(congestion_score / 100) * sensor_noise_stress` 등 비선형 결합 |
| 출력 정책 | 제출물 단일 CSV (`submission_v6_*.csv`) |

v6 특징:
- v5의 모든 파생 피처 유지하면서 Permutation으로 중요도 낮은 피처 제거
- 모델 복잡도 감소 → 테스트셋 일반화 성능 향상
- 실행 시간 단축

v6 생성 결과:
- `submission_v6_*.csv`

### Step 14 — v7 (최종 최적화: PCA + Cyclic Time + Permutation)

핵심 목표: **시간/공간 고도화 + 극도의 피처 효율화 (Top-110)**

| 항목 | 내용 |
|---|---|
| 모델 엔진 | `CatBoostRegressor` 단일 모델 (Deep Squeeze 적용) |
| 손실함수 | `loss_function="Quantile:alpha=0.52"`, `eval_metric="MAE"` |
| 로그 변환 | `USE_LOG_TRANSFORM=True` (Long-tail 안정화) |
| Layout 잠재인수 | Layout 물리 피처를 PCA(`n_components=3`) 압축 → `layout_pca_1`, `layout_pca_2`, `layout_pca_3` |
| 시간 특성화 | Cyclic Time Encoding: scenario 내 15분 슬롯을 sin/cos로 변환 |
| 피처 다이어트 | Permutation Importance 기반 `TOP_N_FEATURES=110` (PERM_CANDIDATE=220) |
|  | 1차 학습 → Permutation 계산(`PERM_SAMPLE_SIZE=25000`) → 2차 재학습 |
| 클러스터링 | K-Means(k=10) 유지 |
| 출력 정책 | 제출물 단일 CSV (`submission_v7_*.csv`) |

v7 신규 기능:
- `add_layout_pca()`: PCA로 layout 물리 피처 차원 축약
- Cyclic Time Encoding: 시간 정보를 원형 좌표로 표현
- Permutation Importance 2단계 재학습: 최고 효율의 피처셋 구성

v7 생성 결과:
- `submission_v7_*.csv`

---

## 3. 전체 실행 파이프라인 (v7 최신)

```
데이터 로드
  ↓
layout_info Left Join (layout_id 기준)
  ↓
Layout PCA 압축 (n_components=3) → layout_pca_1, layout_pca_2, layout_pca_3
  ↓
물리·운영 파생 피처 + Ratio/HRI/통신/환경 상호작용 피처 생성
  ↓
Cyclic Time Encoding (sin/cos 변환)
  ↓
Lag(1,2) / Diff(1) 피처
  ↓
Rolling(3,4,5) Mean / Max / Std 피처
  ↓
시나리오 통계 피처 (scenario_id 기준 mean/std)
  ↓
K-Means 레이아웃 클러스터링 (layout_cluster_id, k=10)
  ↓
결측치 플래그 생성 → 시나리오 내 ffill/bfill → 중앙값 fallback
  ↓
카테고리 피처 선언 (layout_cluster_id, layout_type_enc)
  ↓
[1차 학습] GroupKFold 5-fold + CatBoost 
  · 타깃 log1p 변환 / 예측 expm1 복원
  · loss_function = Quantile(alpha=0.52), eval_metric = MAE
  ↓
Permutation Importance 계산 (25,000 샘플 기준)
  ↓
Top-110 피처만 선별
  ↓
[2차 학습] 선별된 피처로 재학습
  ↓
OOF MAE 출력
  ↓
submission_v7_YYYYMMDD_HHMMSS.csv 저장 (data/submission/)
```

---

## 4. 실행 명령어

```bash
# EDA 시각화
python src/eda_visualize.py

# v1 학습
python src/baseline_lightgbm.py

# v2 학습 (layout_info 포함 전체 파이프라인)
python src/baseline_lightgbm_v2.py

# v3 학습 (시계열 피처 강화)
python src/baseline_lightgbm_v3.py

# v4 학습 (클러스터 기반 TE + HRI/통신)
python src/baseline_lightgbm_v4.py

# v5 학습 (CatBoost 전환, 일반화)
python src/baseline_catboost_v5.py

# v6 학습 (피처 다이어트 + 상호작용)
python src/baseline_catboost_v6.py

# v7 학습 (최신: PCA + Cyclic Time + Permutation)
python src/baseline_catboost_v7.py

# 중요도 시각화 (분리 실행)
python src/report_importance.py --csv reports/eda/feature_importance_v3.csv
```

---

## 5. 제출 파일 이력

| 파일 | 버전 | 전략 | 상태 |
|---|---|---|---|
| `submission.csv` | v1 | Log + GroupKFold 기본 | ✅ |
| `submission_v2.csv` | v2 | Layout 통합 | ✅ |
| `submission_v3_*.csv` | v3 | 시계열 피처 | ✅ |
| `submission_v4_*.csv` | v4 | 클러스터 TE + HRI | ✅ |
| `submission_v5_*.csv` | v5 | CatBoost 단일 | ✅ |
| `submission_v6_*.csv` | v6 | 피처 다이어트 | ✅ |
| `submission_v7_*.csv` | v7 | PCA + Cyclic Time + Permutation | 🔄 (코드 완성, 실행 예정) |

---

## 6. GitHub 업로드 이력

- 원격 저장소: `jinsan02/26_04_monthly_dacon` (private)
- 초기 푸시 중 `train.csv`(100MB 초과) 문제 발생
- 해결:
  - `.gitignore`로 `data/raw/*.csv`, `data/meta/*.csv` 제외
  - Git 히스토리 정리 후 코드/리포트 중심으로 재커밋
- 현재 방향: **데이터 파일 제외 + 코드/문서/리포트만 버전관리**
- 최신 커밋 (2026-04-26): v3~v7 문서화 추가

---

## 7. 향후 고려 사항

- **XGBoost 추가 앙상블**: v7 CatBoost + 기존 LightGBM(v2/v4) + XGBoost 가중 평균 비교
- **가중 평균 앙상블**: LightGBM 0.4 : CatBoost 0.3 : XGBoost 0.3 비율
- **Optuna HPO**: `build_lgbm()` 파라미터를 trial 객체로 교체하면 즉시 튜닝 가능
- **학습률 추가 조정**: v7의 `learning_rate`, `n_estimators`, `quantile alpha` 추가 최적화
