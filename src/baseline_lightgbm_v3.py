"""Smart warehouse delay prediction — v4 (일반화 특화).

V4 핵심 개선 사항 (새로운 레이아웃 일반화 목표):
------------------------------------------------------
전략 1: 레이아웃 클러스터 기반 Target Encoding
   - layout_id_target_enc 제거 → layout_cluster_target_enc로 대체
   - K-Means 클러스터(10개) 단위로 Smoothed Fold-safe 인코딩
   - 처음 보는 layout_id에도 '유사 구조 창고' 평균 지연값 적용

전략 2: 물리적 병목 피처 강화
   - path_complexity: (floor_area_sqm / obstacle_ratio) / rack_count
     로봇이 랙 사이를 빠져나갈 때 겪을 물리적 복잡도
   - congestion_persistence: 최근 3 슬롯 혼잡도 표준편차
     병목이 '지속되는 상황'을 수치화

전략 3: 모델 학습 조건 미세 조정
   - objective=quantile(alpha=0.55): MAE의 수학적 목표(중잇값)를 직접 공략
     alpha=0.55로 약간 상향 예측 (Long-tail 분포 보정)
   - colsample_bytree=0.7: layout_cluster_target_enc 의존도 강제 분산

V3 유지 사항:
   - Smoothed Target Encoding (alpha=10)
   - time_idx, cumulative_inflow 시계열 피처
   - GroupKFold by scenario_id
   - lag/rolling/interaction 피처
   - is_missing 플래그 pruning
   - 시각화는 report_importance.py에서 별도 실행
"""

from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------
TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
N_SPLITS = 5

LAG_COLS = [
    "order_inflow_15m", "congestion_score",
    "low_battery_ratio", "battery_mean",
]
ROLLING_COLS = [
    "order_inflow_15m", "congestion_score",
    "low_battery_ratio", "robot_active", "battery_mean",
]
SCENARIO_STAT_COLS = [
    "order_inflow_15m", "congestion_score",
    "low_battery_ratio", "robot_active", "battery_mean",
]
ROLLING_WINDOWS = [3, 4, 5]

# Smoothed target encoding alpha (클수록 전체 평균 쪽으로 수렴)
TARGET_ENC_ALPHA = 10.0

# 레이아웃 클러스터 수 (클러스터 기반 Target Encoding에 사용)
N_LAYOUT_CLUSTERS = 10

# True: log1p(y) 학습 / False: 원본 MAE 직접 최적화
USE_LOG_TRANSFORM = True

# quantile regression alpha (0.5=중잇값, 0.55=약간 상향 예측)
QUANTILE_ALPHA = 0.55

# is_missing 피처 fold-평균 중요도 하한
MISSING_IMP_THRESHOLD = 10.0



# ---------------------------------------------------------------------------
# Layout Info Integration
# ---------------------------------------------------------------------------

def merge_layout_info(
    train: pd.DataFrame, test: pd.DataFrame, layout_info: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    li = layout_info.copy()

    li["floor_area_per_robot"] = li["floor_area_sqm"] / (li["robot_total"] + 1)
    li["intersection_density"] = li["intersection_count"] / (li["floor_area_sqm"] + 1)
    li["aisle_compactness"] = li["layout_compactness"] / (li["aisle_width_avg"] + 1e-3)
    li["route_constraint"] = li["one_way_ratio"] * li["intersection_count"]
    li["charger_ratio"] = li["robot_total"] / (li["charger_count"] + 1)
    li["pack_station_per_area"] = li["pack_station_count"] / (li["floor_area_sqm"] + 1)
    li["layout_type_enc"] = li["layout_type"].astype("category").cat.codes

    # [v4] 물리적 경로 복잡도: 로봇이 랙 사이를 빠져나갈 때 겪는 복잡도 수치화
    # (area / obstacle_ratio) / rack_count — 클수록 복잡도가 낮아 이동이 쉬움
    if {"floor_area_sqm", "obstacle_ratio", "rack_count"}.issubset(li.columns):
        li["path_complexity"] = (
            li["floor_area_sqm"] / (li["obstacle_ratio"].replace(0, np.nan).fillna(1e-3))
        ) / (li["rack_count"] + 1)

    train = train.merge(li.drop(columns=["layout_type"]), on="layout_id", how="left")
    test = test.merge(li.drop(columns=["layout_type"]), on="layout_id", how="left")
    return train, test


def fit_layout_cluster(
    layout_info: pd.DataFrame,
    n_clusters: int = N_LAYOUT_CLUSTERS,
) -> tuple[KMeans, StandardScaler, list[str]]:
    """layout_info 물리 피처로 K-Means 학습. 반환값은 (model, scaler, used_cols)."""
    cluster_cols = [
        "floor_area_sqm", "rack_count", "obstacle_ratio",
        "aisle_count", "aisle_width_avg", "layout_compactness",
        "zone_dispersion", "intersection_count", "one_way_ratio",
        "charger_count", "pack_station_count", "robot_total",
    ]
    available = [c for c in cluster_cols if c in layout_info.columns]
    scaler = StandardScaler()
    X = scaler.fit_transform(layout_info[available].fillna(0))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X)
    return km, scaler, available


def add_layout_cluster(
    train: pd.DataFrame,
    test: pd.DataFrame,
    layout_info: pd.DataFrame,
    n_clusters: int = N_LAYOUT_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """layout_info 기반 K-Means 클러스터를 layout_cluster_id로 부여.

    클러스터링은 layout_info(정적 테이블) 기준으로만 수행하므로
    - train/test에 없는 레이아웃도 올바르게 처리됨
    - layout_cluster_id가 전략 1 Target Encoding의 핵심 키가 됨
    """
    km, scaler, used_cols = fit_layout_cluster(layout_info, n_clusters)

    # layout_info에 클러스터 id 부여 후 조인
    li_cluster = layout_info[["layout_id"] + used_cols].copy()
    li_cluster["layout_cluster_id"] = km.predict(
        scaler.transform(li_cluster[used_cols].fillna(0))
    )

    train = train.merge(li_cluster[["layout_id", "layout_cluster_id"]], on="layout_id", how="left")
    test = test.merge(li_cluster[["layout_id", "layout_cluster_id"]], on="layout_id", how="left")

    # test에 layout_cluster_id가 없으면 nearest cluster 추론
    if test["layout_cluster_id"].isna().any() and used_cols:
        avail_test = [c for c in used_cols if c in test.columns]
        if avail_test:
            missing_mask = test["layout_cluster_id"].isna()
            test.loc[missing_mask, "layout_cluster_id"] = km.predict(
                scaler.transform(test.loc[missing_mask, avail_test].fillna(0))
            )

    return train, test


def add_layout_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: pd.Series,
    alpha: float = TARGET_ENC_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[전략 1] 클러스터 기반 Smoothed Fold-safe Target Encoding.

    layout_id 대신 layout_cluster_id 단위로 인코딩하여
    처음 보는 layout_id에도 '유사 구조 창고' 평균 지연값을 적용할 수 있게 함.

    Smoothing formula:
        enc = (n * fold_mean + alpha * global_mean) / (n + alpha)
    """
    enc_col = "layout_cluster_target_enc"
    group_key = "layout_cluster_id" if "layout_cluster_id" in train.columns else "layout_id"

    global_mean = train[TARGET].mean()
    train[enc_col] = np.nan
    gkf = GroupKFold(n_splits=N_SPLITS)

    for _, val_idx in gkf.split(train, train[TARGET], groups=groups):
        tr_mask = train.index.difference(train.index[val_idx])
        fold_stats = (
            train.loc[tr_mask]
            .groupby(group_key)[TARGET]
            .agg(["count", "mean"])
        )
        fold_stats["smoothed"] = (
            (fold_stats["count"] * fold_stats["mean"] + alpha * global_mean)
            / (fold_stats["count"] + alpha)
        )
        train.iloc[
            val_idx, train.columns.get_loc(enc_col)
        ] = train.iloc[val_idx][group_key].map(fold_stats["smoothed"])

    train[enc_col] = train[enc_col].fillna(global_mean)

    # test: 전체 train 기반 smoothed 값
    full_stats = train.groupby(group_key)[TARGET].agg(["count", "mean"])
    full_stats["smoothed"] = (
        (full_stats["count"] * full_stats["mean"] + alpha * global_mean)
        / (full_stats["count"] + alpha)
    )
    test[enc_col] = test[group_key].map(full_stats["smoothed"])

    unknown_mask = test[enc_col].isna()
    if unknown_mask.sum() > 0:
        unknown_clusters = test.loc[unknown_mask, group_key].unique()
        print(f"[target_enc] test에서 매핑 안된 {group_key} {len(unknown_clusters)}개 "
              f"→ global_mean({global_mean:.4f})으로 대체")
        test[enc_col] = test[enc_col].fillna(global_mean)

    return train, test


def add_operational_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    new: dict[str, pd.Series] = {}
    if {"robot_active", "charger_count"}.issubset(df.columns):
        new["charger_load"] = df["robot_active"] / (df["charger_count"] + 1)
    if {"order_inflow_15m", "pack_station_count"}.issubset(df.columns):
        new["inflow_per_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1)
    if {"robot_active", "floor_area_per_robot"}.issubset(df.columns):
        new["active_area_ratio"] = df["floor_area_per_robot"] / (df["robot_active"] + 1)
    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    new: dict[str, pd.Series] = {}
    if {"robot_charging", "robot_active"}.issubset(df.columns):
        new["charging_per_active"] = df["robot_charging"] / (df["robot_active"] + 1)
    if {"low_battery_ratio", "battery_mean"}.issubset(df.columns):
        new["low_battery_x_battery_mean"] = df["low_battery_ratio"] * df["battery_mean"]
    if {"order_inflow_15m", "congestion_score"}.issubset(df.columns):
        new["inflow_x_congestion"] = df["order_inflow_15m"] * df["congestion_score"]
    if {"battery_mean", "congestion_score"}.issubset(df.columns):
        new["battery_stress"] = df["battery_mean"] / (df["congestion_score"] + 1)
    if {"battery_mean", "low_battery_ratio"}.issubset(df.columns):
        new["battery_risk"] = (1 - df["battery_mean"]) * df["low_battery_ratio"]
    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def add_congestion_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """[전략 2] 병목 지속 지표: 최근 3 슬롯 congestion_score 표준편차.

    단순 전 시점 값보다 '혼잡이 얼마나 지속되는지'를 포착하여
    새로운 레이아웃에서도 유효한 범용 물리 신호로 작동.
    """
    if "scenario_id" not in df.columns or "congestion_score" not in df.columns:
        return df

    work = df.sort_values(_get_sort_keys(df)).copy()
    shifted = work.groupby("scenario_id")["congestion_score"].shift(1)
    work["congestion_persistence"] = (
        shifted
        .groupby(work["scenario_id"])
        .transform(lambda x: x.rolling(3, min_periods=2).std())
    )
    # NaN(초반 슬롯) → 0으로 채워 안정적으로 관리
    work["congestion_persistence"] = work["congestion_persistence"].fillna(0)
    return work.sort_index()


def _get_sort_keys(df: pd.DataFrame) -> list[str]:
    sort_keys = ["scenario_id"]
    for candidate in ["time_idx", "time_step", "timeslot", "timestamp", "ID"]:
        if candidate in df.columns:
            sort_keys.append(candidate)
            break
    return sort_keys


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시나리오 내 타임슬롯 순서(time_idx, 0~24)와 누적 주문량을 추가.

    - time_idx: 시나리오 후반부일수록 지연 누적 경향 반영
    - cumulative_inflow: 창고 피로도 측정 (현재까지 쌓인 주문 부담)
    """
    if "scenario_id" not in df.columns:
        return df

    sort_keys = _get_sort_keys(df)
    work = df.sort_values(sort_keys).copy()

    # time_idx: 시나리오 내 행 순번 (0-based)
    work["time_idx"] = work.groupby("scenario_id").cumcount()

    # cumulative_inflow: 시나리오 내 order_inflow_15m 누적합 (현재 포함)
    if "order_inflow_15m" in work.columns:
        work["cumulative_inflow"] = (
            work.groupby("scenario_id")["order_inflow_15m"].cumsum()
        )
        # 이전 슬롯까지의 누적 (현재 제외, leakage 방지)
        work["cumulative_inflow_lag1"] = work.groupby("scenario_id")["cumulative_inflow"].shift(1).fillna(0)

    return work.sort_index()


def add_lag_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    if "scenario_id" not in df.columns:
        return df

    work = df.sort_values(_get_sort_keys(df)).copy()
    for col in base_cols:
        if col not in work.columns:
            continue
        grp = work.groupby("scenario_id")[col]
        work[f"{col}_lag1"] = grp.shift(1)
        work[f"{col}_lag2"] = grp.shift(2)
        work[f"{col}_diff1"] = work[col] - work[f"{col}_lag1"]
    return work.sort_index()


def add_rolling_features(
    df: pd.DataFrame, base_cols: list[str], windows: list[int]
) -> pd.DataFrame:
    if "scenario_id" not in df.columns:
        return df

    work = df.sort_values(_get_sort_keys(df)).copy()
    for col in base_cols:
        if col not in work.columns:
            continue
        for w in windows:
            grp = work.groupby("scenario_id")[col]
            shifted = grp.shift(1)
            work[f"{col}_roll{w}_mean"] = shifted.groupby(work["scenario_id"]).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            work[f"{col}_roll{w}_max"] = shifted.groupby(work["scenario_id"]).transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )
            work[f"{col}_roll{w}_std"] = shifted.groupby(work["scenario_id"]).transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
    return work.sort_index()


def add_scenario_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "scenario_id" not in train_df.columns:
        return train_df, test_df
    for col in stat_cols:
        if col not in train_df.columns:
            continue
        grp = train_df.groupby("scenario_id")[col].agg(["mean", "std"])
        train_df[f"{col}_scenario_mean"] = train_df["scenario_id"].map(grp["mean"])
        train_df[f"{col}_scenario_std"] = train_df["scenario_id"].map(grp["std"])
        test_df[f"{col}_scenario_mean"] = test_df["scenario_id"].map(grp["mean"])
        test_df[f"{col}_scenario_std"] = test_df["scenario_id"].map(grp["std"])
    return train_df, test_df


def add_missing_flags(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_cols = train_df.columns[train_df.isna().mean() > 0]
    train_flags: dict[str, pd.Series] = {}
    test_flags: dict[str, pd.Series] = {}
    for col in missing_cols:
        if col == TARGET:
            continue
        flag_col = f"{col}_is_missing"
        train_flags[flag_col] = train_df[col].isna().astype("int8")
        if col in test_df.columns:
            test_flags[flag_col] = test_df[col].isna().astype("int8")
    if train_flags:
        train_df = pd.concat([train_df, pd.DataFrame(train_flags, index=train_df.index)], axis=1)
    if test_flags:
        test_df = pd.concat([test_df, pd.DataFrame(test_flags, index=test_df.index)], axis=1)
    return train_df, test_df


def prune_missing_flags(
    feature_cols: list[str],
    importance: pd.Series,
    threshold: float = MISSING_IMP_THRESHOLD,
) -> list[str]:
    low = [
        c for c in feature_cols
        if c.endswith("_is_missing") and importance.get(c, 0) < threshold
    ]
    if low:
        print(f"[prune] 제거된 is_missing 피처 {len(low)}개: "
              f"{low[:5]}{'...' if len(low) > 5 else ''}")
    return [c for c in feature_cols if c not in low]


def impute_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = train_df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    if "scenario_id" in train_df.columns:
        train_df[numeric_cols] = train_df.groupby("scenario_id")[numeric_cols].transform(
            lambda g: g.ffill().bfill()
        )
    if "scenario_id" in test_df.columns:
        avail = [c for c in numeric_cols if c in test_df.columns]
        test_df[avail] = test_df.groupby("scenario_id")[avail].transform(
            lambda g: g.ffill().bfill()
        )
    medians = train_df[numeric_cols].median()
    train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(medians)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        # [전략 3] quantile(alpha=0.55): MAE의 수학적 목표(중잇값)를 직접 공략
        # Long-tail 분포에서 상향 예측을 약간 허용해 큰 지연 포착
        objective="quantile",
        alpha=QUANTILE_ALPHA,
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        # [전략 3] 0.7로 낮춰 layout_cluster_target_enc 의존도 강제 분산
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _transform_target(y: pd.Series) -> np.ndarray:
    if USE_LOG_TRANSFORM:
        return np.log1p(y.clip(lower=0).values)
    return y.clip(lower=0).values


def _inverse_target(pred: np.ndarray) -> np.ndarray:
    if USE_LOG_TRANSFORM:
        return np.clip(np.expm1(pred), a_min=0, a_max=None)
    return np.clip(pred, a_min=0, a_max=None)


def run_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    groups = train["scenario_id"] if "scenario_id" in train.columns else None
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_preds = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)
    importance_sum = pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(train, train[TARGET], groups=groups), start=1
    ):
        print(f"--- Fold {fold} ---")
        X_tr = train.loc[tr_idx, feature_cols]
        X_val = train.loc[val_idx, feature_cols]
        y_tr = _transform_target(train.loc[tr_idx, TARGET])
        y_val = _transform_target(train.loc[val_idx, TARGET])

        model = build_lgbm()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)],
        )

        oof_preds[val_idx] = _inverse_target(model.predict(X_val))
        test_preds += _inverse_target(model.predict(test[feature_cols])) / N_SPLITS
        importance_sum += pd.Series(model.feature_importances_, index=feature_cols)

    return oof_preds, test_preds, importance_sum / N_SPLITS


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    layout_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # 1. layout_info 조인 + 물리 파생 피처 (path_complexity 포함)
    train, test = merge_layout_info(train, test, layout_info)

    # 2. 시계열 / 상호작용 피처
    for df_name, df in [("train", train), ("test", test)]:
        df = add_time_features(df)
        df = add_interaction_features(df)
        df = add_congestion_persistence(df)   # [전략 2] 병목 지속 지표
        df = add_operational_efficiency_features(df)
        df = add_lag_features(df, LAG_COLS)
        df = add_rolling_features(df, ROLLING_COLS, ROLLING_WINDOWS)
        if df_name == "train":
            train = df
        else:
            test = df

    # 3. 시나리오 통계
    train, test = add_scenario_stats(train, test, SCENARIO_STAT_COLS)

    # 4. [전략 1] layout_info 기반 K-Means 클러스터 (layout_cluster_id 생성)
    train, test = add_layout_cluster(train, test, layout_info, n_clusters=N_LAYOUT_CLUSTERS)

    # 5. Smoothed Fold-safe 타깃 인코딩
    groups = train["scenario_id"] if "scenario_id" in train.columns else None
    train, test = add_layout_target_encoding(train, test, groups, alpha=TARGET_ENC_ALPHA)

    # 6. 결측치 플래그 + 보간
    train, test = add_missing_flags(train, test)
    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    train, test = impute_features(train, test, feature_cols)

    feature_cols = [c for c in feature_cols if c in test.columns]
    feature_cols = train[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()

    for col in ["layout_cluster_id", "layout_type_enc"]:
        if col in train.columns:
            train[col] = train[col].astype("category")
        if col in test.columns:
            test[col] = test[col].astype("category")

    return train, test, feature_cols


# ---------------------------------------------------------------------------
# Main (submission 출력만 담당)
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"
    layout_info_path = project_root / "data" / "meta" / "layout_info.csv"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_root / "data" / "submission" / f"submission_v4_{ts}.csv"
    importance_path = project_root / "reports" / "eda" / "feature_importance_v3.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    layout_info = pd.read_csv(layout_info_path)
    print(f"학습 데이터: {train.shape}  테스트 데이터: {test.shape}  레이아웃: {layout_info.shape}")
    print(f"log transform: {USE_LOG_TRANSFORM} / target_enc_alpha: {TARGET_ENC_ALPHA} / quantile_alpha: {QUANTILE_ALPHA}")

    train, test, feature_cols = build_features(train, test, layout_info)
    print(f"최종 피처 수: {len(feature_cols)}")

    # 1차 학습
    oof_preds, test_preds, importance = run_cv(train, test, feature_cols)

    # 저중요도 is_missing 제거 후 재학습
    pruned_cols = prune_missing_flags(feature_cols, importance)
    if len(pruned_cols) < len(feature_cols):
        print(f"\n[재학습] 피처 {len(feature_cols)} → {len(pruned_cols)}개")
        oof_preds, test_preds, importance = run_cv(train, test, pruned_cols)
        feature_cols = pruned_cols

    oof_mae = mean_absolute_error(train[TARGET], oof_preds)
    print(f"\nOOF MAE: {oof_mae:.6f}")

    # 피처 중요도 CSV 저장 (시각화는 report_importance.py 에서 별도 실행)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance.sort_values(ascending=False).to_csv(importance_path, header=["importance"])
    print(f"피처 중요도 저장: {importance_path.name}")

    # Submission 저장
    submission = pd.DataFrame({"ID": test["ID"], TARGET: test_preds})
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"submission 저장 완료: {submission_path}")
    print("\n시각화: python src/report_importance.py --csv reports/eda/feature_importance_v3.csv")


if __name__ == "__main__":
    main()
