"""Improved baseline v2 for smart warehouse delay prediction.

Changes from v1:
- Rolling window features (mean, max, std over 3~4 timesteps)
- Lower learning_rate=0.01 with higher n_estimators=3000
- Modular design ready for multi-model ensemble

Changes from v1 (layout_info integration):
- Left join layout_info on layout_id
- Physical density / complexity features
- Operational efficiency features (layout × train)
- K-Means layout clustering (layout_group)
- Fold-safe target encoding per layout_id

Insight-driven updates (feature importance analysis):
- battery_mean / (congestion_score + 1): 배터리 부족 + 혼잡 복합 상태 코드화
- low_battery_ratio_lag1 등 lag 피처 중요도 상위 반영 (lag/rolling 유지)
- 중요도 하위 is_missing 플래그 자동 제거 (MISSING_IMP_THRESHOLD 기준)
"""

from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
N_SPLITS = 5

LAG_COLS = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "battery_mean"]
ROLLING_COLS = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "robot_active", "battery_mean"]
SCENARIO_STAT_COLS = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "robot_active", "battery_mean"]
ROLLING_WINDOWS = [3, 4]

# is_missing 플래그 중 fold-평균 중요도가 이 값 미만이면 제거로 모델 경량화
MISSING_IMP_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Layout Info Integration
# ---------------------------------------------------------------------------

def merge_layout_info(train: pd.DataFrame, test: pd.DataFrame, layout_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left join layout_info and add physical/operational derived features."""
    layout_info = layout_info.copy()

    # --- 물리적 밀집도 / 공간 특성 ---
    # 로봇 1대당 담당 면적 (클수록 넓어서 여유있음)
    layout_info["floor_area_per_robot"] = layout_info["floor_area_sqm"] / (layout_info["robot_total"] + 1)
    # 교차로 밀도: 면적 대비 교차로 수 (높을수록 경로 복잡)
    layout_info["intersection_density"] = layout_info["intersection_count"] / (layout_info["floor_area_sqm"] + 1)
    # 통로 폭 대비 레이아웃 밀집도 (좁고 빽빽할수록 높음)
    layout_info["aisle_compactness"] = layout_info["layout_compactness"] / (layout_info["aisle_width_avg"] + 1e-3)
    # 일방통행 비율 × 교차로 수: 경로 제약 복합 지표
    layout_info["route_constraint"] = layout_info["one_way_ratio"] * layout_info["intersection_count"]
    # 충전소 가용 비율: charger 1개당 담당 로봇 수 (높을수록 부족)
    layout_info["charger_ratio"] = layout_info["robot_total"] / (layout_info["charger_count"] + 1)
    # 패킹 스테이션 가용 비율
    layout_info["pack_station_per_area"] = layout_info["pack_station_count"] / (layout_info["floor_area_sqm"] + 1)

    # layout_type 라벨 인코딩 (LightGBM에 전달)
    layout_info["layout_type_enc"] = layout_info["layout_type"].astype("category").cat.codes

    train = train.merge(layout_info.drop(columns=["layout_type"]), on="layout_id", how="left")
    test = test.merge(layout_info.drop(columns=["layout_type"]), on="layout_id", how="left")
    return train, test


def add_layout_cluster(train: pd.DataFrame, test: pd.DataFrame, n_clusters: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-Means clustering on layout static features → layout_group."""
    cluster_cols = [
        "floor_area_per_robot", "intersection_density", "aisle_compactness",
        "route_constraint", "charger_ratio", "layout_compactness", "zone_dispersion",
    ]
    available = [c for c in cluster_cols if c in train.columns]
    if not available:
        return train, test

    scaler = StandardScaler()
    X_layout = scaler.fit_transform(train[available].fillna(0))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train["layout_group"] = km.fit_predict(X_layout)
    test["layout_group"] = km.predict(scaler.transform(test[available].fillna(0)))
    return train, test


def add_layout_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-safe target encoding for layout_id using GroupKFold."""
    if "layout_id" not in train.columns:
        return train, test

    enc_col = "layout_id_target_enc"
    train[enc_col] = np.nan
    gkf = GroupKFold(n_splits=N_SPLITS)

    for _, val_idx in gkf.split(train, train[TARGET], groups=groups):
        tr_idx = train.index.difference(train.index[val_idx])
        mean_map = train.loc[tr_idx].groupby("layout_id")[TARGET].mean()
        train.iloc[val_idx, train.columns.get_loc(enc_col)] = (
            train.iloc[val_idx]["layout_id"].map(mean_map)
        )

    # test는 전체 train 기준 평균으로 채움
    global_mean_map = train.groupby("layout_id")[TARGET].mean()
    test[enc_col] = test["layout_id"].map(global_mean_map)

    # fold에서 누락된 경우 전체 평균으로 대체
    global_mean = train[TARGET].mean()
    train[enc_col] = train[enc_col].fillna(global_mean)
    test[enc_col] = test[enc_col].fillna(global_mean)
    return train, test


def add_operational_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runtime features combining train columns with layout-derived columns."""
    new: dict[str, pd.Series] = {}

    # 충전소 부하도: 현재 가동 로봇 / charger_count
    if {"robot_active", "charger_count"}.issubset(df.columns):
        new["charger_load"] = df["robot_active"] / (df["charger_count"] + 1)

    # 통로 혼잡 예상치: 주문 유입량 / pack_station_count
    if {"order_inflow_15m", "pack_station_count"}.issubset(df.columns):
        new["inflow_per_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1)

    # 로봇 가동 대비 넓이: 실제 부하 상태에서의 면적 여유
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

    # 피처 중요도 1위: 배터리 부족 + 혼잡 복합 상태 지표
    # battery가 낙싙수록 & congestion이 높을수록 지연 위험 증가
    if {"battery_mean", "congestion_score"}.issubset(df.columns):
        new["battery_stress"] = df["battery_mean"] / (df["congestion_score"] + 1)

    # 배터리 잔량과 저배터리 비율의 상호작용: 심각한 복합 첨피 지표
    if {"battery_mean", "low_battery_ratio"}.issubset(df.columns):
        new["battery_risk"] = (1 - df["battery_mean"]) * df["low_battery_ratio"]

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def _get_sort_keys(df: pd.DataFrame) -> list[str]:
    sort_keys = ["scenario_id"]
    for candidate in ["time_idx", "time_step", "timeslot", "timestamp", "ID"]:
        if candidate in df.columns:
            sort_keys.append(candidate)
            break
    return sort_keys


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


def add_rolling_features(df: pd.DataFrame, base_cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Add rolling mean, max, std to capture sustained bottleneck patterns."""
    if "scenario_id" not in df.columns:
        return df

    work = df.sort_values(_get_sort_keys(df)).copy()

    for col in base_cols:
        if col not in work.columns:
            continue
        for w in windows:
            grp = work.groupby("scenario_id")[col]
            # shift(1) to exclude current timestep (avoid target leakage)
            rolled = grp.shift(1).groupby(work["scenario_id"]).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            work[f"{col}_roll{w}_mean"] = rolled

            work[f"{col}_roll{w}_max"] = grp.shift(1).groupby(work["scenario_id"]).transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )

            work[f"{col}_roll{w}_std"] = grp.shift(1).groupby(work["scenario_id"]).transform(
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
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
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
    """is_missing 플래그 중 fold-평균 중요도가 threshold 미만인 것을 제거한 피처 리스트 반환.
    모델을 경량화하면서 고성능 피처만 유지.
    """
    low_importance_flags = [
        c for c in feature_cols
        if c.endswith("_is_missing") and importance.get(c, 0) < threshold
    ]
    if low_importance_flags:
        print(f"[prune] 제거된 is_missing 피처 {len(low_importance_flags)}개: "
              f"{low_importance_flags[:5]}{'...' if len(low_importance_flags) > 5 else ''}")
    return [c for c in feature_cols if c not in low_importance_flags]


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
        available = [c for c in numeric_cols if c in test_df.columns]
        test_df[available] = test_df.groupby("scenario_id")[available].transform(
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
        objective="regression_l1",
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


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
        y_tr = np.log1p(train.loc[tr_idx, TARGET].clip(lower=0))
        y_val = np.log1p(train.loc[val_idx, TARGET].clip(lower=0))

        model = build_lgbm()
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)],
        )

        oof_preds[val_idx] = np.clip(np.expm1(model.predict(X_val)), a_min=0, a_max=None)
        test_preds += np.clip(np.expm1(model.predict(test[feature_cols])), a_min=0, a_max=None) / N_SPLITS
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
    # 1. layout_info 조인 + 물리/정적 파생 피처
    train, test = merge_layout_info(train, test, layout_info)

    # 2. 시계열/상호작용 피처
    for df_name, df in [("train", train), ("test", test)]:
        df = add_interaction_features(df)
        df = add_operational_efficiency_features(df)
        df = add_lag_features(df, LAG_COLS)
        df = add_rolling_features(df, ROLLING_COLS, ROLLING_WINDOWS)
        if df_name == "train":
            train = df
        else:
            test = df

    # 3. 시나리오 통계 피처
    train, test = add_scenario_stats(train, test, SCENARIO_STAT_COLS)

    # 4. K-Means 레이아웃 클러스터
    train, test = add_layout_cluster(train, test, n_clusters=6)

    # 5. Fold-safe 타깃 인코딩 (결측치 처리 전에 수행)
    groups = train["scenario_id"] if "scenario_id" in train.columns else None
    train, test = add_layout_target_encoding(train, test, groups)

    # 6. 결측치 플래그 + 보간
    train, test = add_missing_flags(train, test)
    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    train, test = impute_features(train, test, feature_cols)

    feature_cols = [c for c in feature_cols if c in test.columns]
    feature_cols = train[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()

    # LightGBM 카테고리 최적화: layout_group은 정수 category로 선언
    for col in ["layout_group", "layout_type_enc"]:
        if col in train.columns:
            train[col] = train[col].astype("category")
        if col in test.columns:
            test[col] = test[col].astype("category")

    return train, test, feature_cols


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"
    layout_info_path = project_root / "data" / "meta" / "layout_info.csv"

    # 제출 파일명: submission_v2_YYYYMMDD_HHMMSS.csv
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_root / "data" / "submission" / f"submission_v2_{ts}.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    layout_info = pd.read_csv(layout_info_path)
    print(f"학습 데이터: {train.shape}  테스트 데이터: {test.shape}  레이아웃: {layout_info.shape}")

    train, test, feature_cols = build_features(train, test, layout_info)
    print(f"최종 피처 수: {len(feature_cols)}")

    oof_preds, test_preds, importance = run_cv(train, test, feature_cols)

    # 1차 학습 결과로 중요도 낮은 is_missing 피처 제거 후 재학습
    pruned_cols = prune_missing_flags(feature_cols, importance)
    if len(pruned_cols) < len(feature_cols):
        print(f"\n[재학습] 피처 {len(feature_cols)} → {len(pruned_cols)}개로 경량화 후 재학습")
        oof_preds, test_preds, importance = run_cv(train, test, pruned_cols)
        feature_cols = pruned_cols

    oof_mae = mean_absolute_error(train[TARGET], oof_preds)
    print(f"OOF MAE: {oof_mae:.6f}")

    # --- 피처 중요도 시각화 저장 ---
    save_feature_importance(importance, project_root / "reports" / "eda")

    submission = pd.DataFrame({"ID": test["ID"], TARGET: test_preds})
    submission.to_csv(submission_path, index=False)
    print(f"submission 저장 완료: {submission_path}")


# ---------------------------------------------------------------------------
# Feature Importance Visualization
# ---------------------------------------------------------------------------

def save_feature_importance(
    importance: pd.Series,
    out_dir: Path,
    top_n: int = 40,
) -> None:
    """Fold-averaged gain importance → PNG + CSV. 이미 파일이 있으면 겹어쓰기 스킵."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top = importance.sort_values(ascending=False).head(top_n)

    # --- 전체 Top-N 막대 그래프 ---
    p_top = out_dir / "06_feature_importance_top40.png"
    if not p_top.exists():
        fig, ax = plt.subplots(figsize=(12, max(6, top_n // 3)))
        top[::-1].plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Feature Importance (Top {top_n}, fold-averaged gain)")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(p_top, dpi=180)
        plt.close(fig)
        print(f"[saved] {p_top.name}")
    else:
        print(f"[skip]  {p_top.name} 이미 존재")

    # --- 레이아웃 파생 피처만 따로 강조 ---
    layout_keys = [
        "floor_area_per_robot", "intersection_density", "aisle_compactness",
        "route_constraint", "charger_ratio", "pack_station_per_area",
        "charger_load", "inflow_per_station", "active_area_ratio",
        "layout_group", "layout_id_target_enc", "layout_type_enc",
        "battery_stress", "battery_risk",
    ]
    layout_imp = importance[importance.index.isin(layout_keys)].sort_values(ascending=False)
    if not layout_imp.empty:
        p_layout = out_dir / "07_layout_feature_importance.png"
        if not p_layout.exists():
            fig2, ax2 = plt.subplots(figsize=(10, max(4, len(layout_imp) // 2)))
            layout_imp[::-1].plot(kind="barh", ax=ax2, color="darkorange")
            ax2.set_title("Layout Feature Importance (fold-averaged gain)")
            ax2.set_xlabel("Importance")
            ax2.set_ylabel("Feature")
            plt.tight_layout()
            fig2.savefig(p_layout, dpi=180)
            plt.close(fig2)
            print(f"[saved] {p_layout.name}")
        else:
            print(f"[skip]  {p_layout.name} 이미 존재")

    # CSV는 매 실행마다 최신값으로 덮어쓰기 (수치 추적 목적)
    csv_path = out_dir / "feature_importance.csv"
    importance.sort_values(ascending=False).to_csv(csv_path, header=["importance"])
    print(f"[saved] {csv_path.name} (업데이트)")

    print(f"\n[Feature Importance] layout 피처 순위:")
    print(layout_imp.to_string())


if __name__ == "__main__":
    main()
