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
│       └── submission_v2.csv       # v2 제출 결과 ✅
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
│       └── feature_importance.csv              # 전체 피처 중요도 수치 ✅
├── src/
│   ├── baseline_lightgbm.py        # v1 개선 베이스라인
│   ├── baseline_lightgbm_v2.py     # v2 최종 코드 (layout 통합)
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

---

## 3. 전체 실행 파이프라인 (v2)

```
데이터 로드
  ↓
layout_info Left Join (layout_id 기준)
  ↓
물리·운영 파생 피처 생성
  ↓
Lag(1,2) / Diff(1) 피처
  ↓
Rolling(3,4) Mean / Max / Std 피처
  ↓
시나리오 통계 피처 (scenario_id 기준 mean/std)
  ↓
K-Means 레이아웃 클러스터링 (layout_group)
  ↓
Fold-safe 타깃 인코딩 (layout_id_target_enc)
  ↓
결측치 플래그 생성 → 시나리오 내 ffill/bfill → 중앙값 fallback
  ↓
카테고리 피처 선언 (layout_group, layout_type_enc)
  ↓
GroupKFold 5-fold 학습
  · 타깃 log1p 변환 / 예측 expm1 복원
  · objective = regression_l1 (MAE 직접 최적화)
  · early_stopping patience=150
  ↓
OOF MAE 출력
  ↓
피처 중요도 이미지·CSV 저장 (reports/eda/)
  ↓
submission_v2.csv 저장 (data/submission/)
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
```

---

## 5. 향후 고려 사항

- **CatBoost / XGBoost 앙상블**: `build_lgbm()` 구조를 활용해 모델 빌더 함수만 추가하면 `run_cv()` 재사용 가능
- **가중 평균 앙상블**: LightGBM 0.4 : CatBoost 0.3 : XGBoost 0.3 비율
- **Optuna HPO**: `build_lgbm()` 파라미터를 trial 객체로 교체하면 즉시 튜닝 가능
- **학습률 추가 조정**: `learning_rate=0.01` + `n_estimators=3000` 현재 설정 → 시간 여유 시 0.005로 낮추고 더 높여볼 것
