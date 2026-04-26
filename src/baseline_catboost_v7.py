"""Smart warehouse delay prediction — v7 (Simplify & Interaction++).

핵심 전략:
1) 단일 모델: CatBoostRegressor (Deep Squeeze)
2) 시간/공간 고도화: Cyclic Time + Layout PCA
3) 피처 다이어트: Permutation Importance 기반 Top-N 재학습
4) 앙상블 없이 submission_v7 단일 출력
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
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

# True: log1p(y) 학습 / False: 원본 MAE 직접 최적화
USE_LOG_TRANSFORM = True

# v5는 TE/ID 암기 신호를 제거하되, layout_cluster_id는 유지한다.
DROP_COLS = ["layout_id_target_enc", "layout_cluster_target_enc"]

# 하위 호환: 미사용 함수 기본 인자 평가 시 NameError 방지
TARGET_ENC_ALPHA = 10.0
N_LAYOUT_CLUSTERS = 10

# is_missing 피처 fold-평균 중요도 하한
MISSING_IMP_THRESHOLD = 10.0

# v7 피처 다이어트 목표 (Permutation 기반)
TOP_N_FEATURES = 110
PERM_CANDIDATE_FEATURES = 220
PERM_SAMPLE_SIZE = 25000
PERM_RANDOM_STATE = 42

# PCA 기반 레이아웃 잠재 요인 수
N_LAYOUT_PCA = 3

# Quantile 미세 튜닝 (0.51~0.53 권장 구간)
QUANTILE_ALPHA = 0.52

# v5 체크 대상: HRI/통신/환경 핵심 피처
FOCUS_FEATURE_KEYS = [
    "human_robot_density",
    "low_tenure_risk",
    "staff_x_low_tenure",
    "network_bottleneck",
    "latency_per_active_robot",
    "floor_vibration_idx",
    "ambient_noise_db",
    "sensor_noise_stress",
    "infra_stability_risk",
]


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


def add_layout_pca(
    train: pd.DataFrame,
    test: pd.DataFrame,
    layout_info: pd.DataFrame,
    n_components: int = N_LAYOUT_PCA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """layout_info의 연속형 물리 피처를 PCA로 압축하여 잠재 요인 추가."""
    li = layout_info.copy()
    numeric_cols = li.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return train, test

    scaler = StandardScaler()
    X = scaler.fit_transform(li[numeric_cols].fillna(0))

    n_comp = min(n_components, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    comps = pca.fit_transform(X)

    comp_cols = [f"layout_pca_{i+1}" for i in range(n_comp)]
    pca_df = pd.DataFrame(comps, columns=comp_cols, index=li.index)
    pca_df.insert(0, "layout_id", li["layout_id"].values)

    train = train.merge(pca_df, on="layout_id", how="left")
    test = test.merge(pca_df, on="layout_id", how="left")
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


def add_hri_infra_features(df: pd.DataFrame) -> pd.DataFrame:
    """[v5] HRI + 통신/환경 상호작용 지표를 추가.

    컬럼이 없을 수 있으므로 모든 연산은 조건부로 수행한다.
    """
    new: dict[str, pd.Series] = {}

    # ----------------------------
    # HRI (Human-Robot Interaction)
    # ----------------------------
    if {"order_inflow_15m", "robot_active"}.issubset(df.columns):
        # 운영 압박 지수: 로봇 1대당 실시간 주문 부하
        new["ops_pressure_idx"] = df["order_inflow_15m"] / (df["robot_active"] + 1)

    if {"staff_on_floor", "robot_active"}.issubset(df.columns):
        new["human_robot_density"] = df["staff_on_floor"] / (df["robot_active"] + 1)

    if "worker_avg_tenure_months" in df.columns:
        # 숙련도 부족 지표 (낮을수록 값이 큼)
        new["low_tenure_risk"] = 1 / (df["worker_avg_tenure_months"] + 1)

    if {"staff_on_floor", "worker_avg_tenure_months"}.issubset(df.columns):
        # 인력 많음 + 숙련도 낮음 조합에서 병목 심화 가정
        new["staff_x_low_tenure"] = df["staff_on_floor"] / (df["worker_avg_tenure_months"] + 1)

    if {"congestion_score", "worker_avg_tenure_months"}.issubset(df.columns):
        # 숙련도 가중 혼잡도: 숙련이 낮을수록 혼잡 영향 확대
        new["congestion_tenure_weighted"] = df["congestion_score"] / (
            df["worker_avg_tenure_months"] + 1
        )

    # ----------------------------
    # Network / Infra / Environment
    # ----------------------------
    if {"network_latency_ms", "wifi_signal_db"}.issubset(df.columns):
        # latency↑, wifi(절대세기)↓일수록 통신 병목 증가
        wifi_strength = df["wifi_signal_db"].abs() + 1
        new["network_bottleneck"] = df["network_latency_ms"] / wifi_strength
        # 통신 불안정성: latency와 신호 품질 차이를 단순 수치화
        new["network_instability"] = df["network_latency_ms"] - df["wifi_signal_db"]

    if {"network_latency_ms", "robot_active"}.issubset(df.columns):
        new["latency_per_active_robot"] = df["network_latency_ms"] / (df["robot_active"] + 1)

    if {"floor_vibration_idx", "ambient_noise_db"}.issubset(df.columns):
        # 환경 노이즈 복합 지표
        new["sensor_noise_stress"] = (
            df["floor_vibration_idx"] * np.log1p(df["ambient_noise_db"].clip(lower=0))
        )

    if {
        "network_latency_ms", "floor_vibration_idx", "ambient_noise_db"
    }.issubset(df.columns):
        new["infra_stability_risk"] = (
            df["network_latency_ms"]
            * (1 + df["floor_vibration_idx"]) 
            * np.log1p(df["ambient_noise_db"].clip(lower=0))
        )

    if {"ambient_noise_db", "floor_vibration_idx"}.issubset(df.columns):
        # 환경 스트레스: 소음 + 진동
        new["environment_stress"] = df["ambient_noise_db"] + df["floor_vibration_idx"]

    # v6 핵심: 비선형 폭발형 상호작용 (혼잡 임계 구간을 강조)
    if {"congestion_score", "sensor_noise_stress"}.issubset(df.columns):
        scaled = np.exp(np.clip(df["congestion_score"], -200, 200) / 100.0)
        new["congestion_noise_explosive"] = scaled * df["sensor_noise_stress"]

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """v5 핵심: 절대값 대신 상대 비율로 일반화 강화."""
    new: dict[str, pd.Series] = {}

    if {"robot_active", "floor_area_sqm"}.issubset(df.columns):
        new["robot_active_per_area"] = df["robot_active"] / (df["floor_area_sqm"] + 1)

    if {"order_inflow_15m", "aisle_count"}.issubset(df.columns):
        new["inflow_per_aisle"] = df["order_inflow_15m"] / (df["aisle_count"] + 1)

    if {"congestion_score", "scenario_id"}.issubset(df.columns):
        scen_mean = df.groupby("scenario_id")["congestion_score"].transform("mean")
        new["congestion_vs_scenario_mean"] = df["congestion_score"] / (scen_mean + 1e-6)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 컬럼이 존재하면 Sin/Cos 주기 인코딩 추가."""
    new: dict[str, pd.Series] = {}

    for col, period in [("shift_hour", 24), ("hour", 24), ("time_slot", 24)]:
        if col in df.columns:
            val = df[col].astype(float)
            new[f"{col}_sin"] = np.sin(2 * np.pi * val / period)
            new[f"{col}_cos"] = np.cos(2 * np.pi * val / period)
            break

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def add_short_term_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """30분 혼잡 평균과 주문 변화율(gradient) 추가."""
    if "scenario_id" not in df.columns:
        return df

    work = df.sort_values(_get_sort_keys(df)).copy()

    if "congestion_score" in work.columns:
        shifted = work.groupby("scenario_id")["congestion_score"].shift(1)
        work["congestion_30m_mean"] = shifted.groupby(work["scenario_id"]).transform(
            lambda x: x.rolling(2, min_periods=1).mean()
        )

    if "order_inflow_15m" in work.columns:
        prev = work.groupby("scenario_id")["order_inflow_15m"].shift(1)
        work["order_inflow_gradient"] = (work["order_inflow_15m"] - prev) / (prev.abs() + 1)

    return work.sort_index()


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
    # 컬럼명이 중복되면 train_df[col]이 DataFrame(2D)을 반환할 수 있으므로
    # 고유 컬럼명으로 정리 후 1D Series만 플래그 생성에 사용한다.
    missing_cols = pd.Index(train_df.columns[train_df.isna().mean() > 0]).unique()
    train_flags: dict[str, pd.Series] = {}
    test_flags: dict[str, pd.Series] = {}

    for col in missing_cols:
        if col == TARGET:
            continue
        flag_col = f"{col}_is_missing"

        train_col = train_df[col]
        if isinstance(train_col, pd.DataFrame):
            train_col = train_col.iloc[:, 0]
        train_flags[flag_col] = train_col.isna().astype("int8")

        if col in test_df.columns:
            test_col = test_df[col]
            if isinstance(test_col, pd.DataFrame):
                test_col = test_col.iloc[:, 0]
            test_flags[flag_col] = test_col.isna().astype("int8")

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

def build_lgbm() -> CatBoostRegressor:
    return CatBoostRegressor(
        # v7 Deep Squeeze
        loss_function=f"Quantile:alpha={QUANTILE_ALPHA}",
        eval_metric="MAE",
        iterations=3000,
        learning_rate=0.015,
        depth=6,
        l2_leaf_reg=12.0,
        bootstrap_type="MVS",
        subsample=0.8,
        grow_policy="Lossguide",
        max_leaves=64,
        min_data_in_leaf=64,
        od_type="Iter",
        random_seed=42,
        verbose=300,
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

        cat_cols = X_tr.select_dtypes(include=["category", "object"]).columns.tolist()

        model = build_lgbm()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            cat_features=cat_cols if cat_cols else None,
            use_best_model=True,
            early_stopping_rounds=150,
        )

        oof_preds[val_idx] = _inverse_target(model.predict(X_val))
        test_preds += _inverse_target(model.predict(test[feature_cols])) / N_SPLITS
        importance_sum += pd.Series(model.feature_importances_, index=feature_cols)
    return oof_preds, test_preds, importance_sum / N_SPLITS


def select_top_features(
    feature_cols: list[str],
    importance: pd.Series,
    top_n: int = TOP_N_FEATURES,
) -> list[str]:
    """v6 피처 다이어트: fold 평균 중요도 기반 Top-N 선택.

    - is_missing 플래그는 우선순위를 낮추되, 중요하면 남길 수 있도록 완전 배제는 하지 않음.
    """
    ranked = importance.sort_values(ascending=False)
    selected = ranked.head(top_n).index.tolist()

    # 안전장치: 최소 50개는 유지
    if len(selected) < 50:
        selected = ranked.head(50).index.tolist()

    dropped = [c for c in feature_cols if c not in selected]
    print(f"\n[v6-prune] 피처 다이어트: {len(feature_cols)} -> {len(selected)}")
    if dropped:
        print(f"[v6-prune] 제외 예시: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
    return selected


def permutation_importance_one_fold(
    train: pd.DataFrame,
    feature_cols: list[str],
    max_candidates: int = PERM_CANDIDATE_FEATURES,
    sample_size: int = PERM_SAMPLE_SIZE,
) -> pd.Series:
    """첫 fold 검증셋에서 permutation importance 계산(원본 MAE 기준)."""
    groups = train["scenario_id"] if "scenario_id" in train.columns else None
    gkf = GroupKFold(n_splits=N_SPLITS)
    tr_idx, val_idx = next(gkf.split(train, train[TARGET], groups=groups))

    X_tr = train.loc[tr_idx, feature_cols]
    X_val_full = train.loc[val_idx, feature_cols]
    y_tr = _transform_target(train.loc[tr_idx, TARGET])
    y_val_raw = train.loc[val_idx, TARGET].values

    if len(X_val_full) > sample_size:
        rng = np.random.default_rng(PERM_RANDOM_STATE)
        pick = rng.choice(len(X_val_full), size=sample_size, replace=False)
        X_val = X_val_full.iloc[pick].copy()
        y_val_raw = y_val_raw[pick]
    else:
        X_val = X_val_full.copy()

    cat_cols = X_tr.select_dtypes(include=["category", "object"]).columns.tolist()
    model = build_lgbm()
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, _transform_target(pd.Series(y_val_raw)))],
        cat_features=cat_cols if cat_cols else None,
        use_best_model=True,
        early_stopping_rounds=150,
    )

    base_pred = _inverse_target(model.predict(X_val))
    base_mae = mean_absolute_error(y_val_raw, base_pred)

    gain_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    candidates = gain_imp.head(min(max_candidates, len(gain_imp))).index.tolist()

    rng = np.random.default_rng(PERM_RANDOM_STATE)
    perm_scores: dict[str, float] = {}
    for col in candidates:
        shuffled = X_val.copy()
        shuffled[col] = rng.permutation(shuffled[col].values)
        pred = _inverse_target(model.predict(shuffled))
        mae = mean_absolute_error(y_val_raw, pred)
        perm_scores[col] = mae - base_mae

    return pd.Series(perm_scores).sort_values(ascending=False)


def monitor_focus_feature_importance(
    importance: pd.Series,
    focus_keys: list[str] = FOCUS_FEATURE_KEYS,
    top_n: int = 20,
) -> None:
    """v5 핵심 피처(HRI/통신/환경)의 중요도 순위를 출력한다."""
    ranked = importance.sort_values(ascending=False)
    top_features = ranked.head(top_n).index.tolist()

    present_focus = [k for k in focus_keys if k in ranked.index]
    in_top = [k for k in present_focus if k in top_features]

    print(f"\n[v5-check] 핵심 피처 Top{top_n} 진입: {len(in_top)}/{len(present_focus)}")
    if in_top:
        for k in in_top:
            rank = int(ranked.index.get_loc(k) + 1)
            print(f"  - {k}: rank={rank}, importance={ranked.loc[k]:.2f}")
    else:
        print("  - Top20 진입 피처 없음")

    # Top20 밖 피처는 개선 액션 힌트 제공
    out_top = [k for k in present_focus if k not in top_features]
    if out_top:
        print("[v5-check] Top20 미진입 피처(상호작용 강화 후보):")
        print("  " + ", ".join(out_top))
        print("  힌트: 곱셈/나눗셈 상호작용으로 신호를 증폭해 보세요.")


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

    # 2. 시계열 / ratio / 상호작용 / HRI·인프라 피처
    for df_name, df in [("train", train), ("test", test)]:
        df = add_time_features(df)
        df = add_cyclic_time_features(df)
        df = add_short_term_dynamics(df)
        df = add_ratio_features(df)
        df = add_interaction_features(df)
        df = add_congestion_persistence(df)   # [전략 2] 병목 지속 지표
        df = add_hri_infra_features(df)
        df = add_operational_efficiency_features(df)
        df = add_lag_features(df, LAG_COLS)
        df = add_rolling_features(df, ROLLING_COLS, ROLLING_WINDOWS)
        if df_name == "train":
            train = df
        else:
            test = df

    # 3. 시나리오 통계
    train, test = add_scenario_stats(train, test, SCENARIO_STAT_COLS)

    # 4. 일반화 핵심: layout cluster + layout PCA 잠재 요인
    train, test = add_layout_cluster(train, test, layout_info, n_clusters=15)
    train, test = add_layout_pca(train, test, layout_info, n_components=N_LAYOUT_PCA)

    # 5. 결측치 플래그 + 보간
    train, test = add_missing_flags(train, test)
    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]

    # v5: layout 관련 target encoding / cluster id 완전 제거
    feature_cols = [c for c in feature_cols if c not in DROP_COLS]

    train, test = impute_features(train, test, feature_cols)

    feature_cols = [c for c in feature_cols if c in test.columns]
    # CatBoost는 number/bool/category/object 모두 허용
    feature_cols = [
        c for c in feature_cols
        if (
            pd.api.types.is_numeric_dtype(train[c])
            or pd.api.types.is_bool_dtype(train[c])
            or pd.api.types.is_categorical_dtype(train[c])
            or pd.api.types.is_object_dtype(train[c])
        )
    ]

    for col in ["layout_type_enc", "layout_cluster_id"]:
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
    submission_path = project_root / "data" / "submission" / f"submission_v7_{ts}.csv"
    perm_path = project_root / "reports" / "eda" / "permutation_importance_v7.csv"
    selected_path = project_root / "reports" / "eda" / "selected_features_v7.txt"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    layout_info = pd.read_csv(layout_info_path)
    print(f"학습 데이터: {train.shape}  테스트 데이터: {test.shape}  레이아웃: {layout_info.shape}")
    print(f"log transform: {USE_LOG_TRANSFORM} (v7 default=True)")

    train, test, feature_cols = build_features(train, test, layout_info)
    print(f"최종 피처 수: {len(feature_cols)}")

    # 1차 학습: 전체 피처로 importance 추출
    oof_preds, test_preds, importance = run_cv(train, test, feature_cols)

    # Permutation Importance(1 fold)로 실제 MAE 영향도 측정
    perm_imp = permutation_importance_one_fold(
        train,
        feature_cols,
        max_candidates=PERM_CANDIDATE_FEATURES,
        sample_size=PERM_SAMPLE_SIZE,
    )
    perm_path.parent.mkdir(parents=True, exist_ok=True)
    perm_imp.to_csv(perm_path, header=["mae_increase"])
    print(f"permutation importance 저장 완료: {perm_path}")

    # 2차 학습: permutation 기준 Top-N 선택
    top_features = perm_imp.head(min(TOP_N_FEATURES, len(perm_imp))).index.tolist()
    if len(top_features) < 100:
        # 안전장치: 부족하면 gain importance로 보강
        fill = [c for c in importance.sort_values(ascending=False).index if c not in top_features]
        top_features += fill[: (100 - len(top_features))]

    with open(selected_path, "w", encoding="utf-8") as f:
        for col in top_features:
            f.write(f"{col}\n")
    print(f"선택 피처 저장 완료: {selected_path} ({len(top_features)}개)")

    oof_preds, test_preds, _ = run_cv(train, test, top_features)

    oof_mae = mean_absolute_error(train[TARGET], oof_preds)
    print(f"\nOOF MAE: {oof_mae:.6f}")

    # Submission 저장
    submission = pd.DataFrame({"ID": test["ID"], TARGET: test_preds})
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"submission 저장 완료: {submission_path}")


if __name__ == "__main__":
    main()
