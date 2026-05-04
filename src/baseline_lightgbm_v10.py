"""LightGBM v10: v9 logic + 3-seed ensemble.

Core points:
- Physical mechanics features (saturation, explosion, density, workload, velocity)
- Layout latent factors via PCA(n_components=3)
- GroupKFold by layout_id
- log transform enabled
- Multi-seed ensemble: [42, 10, 2026]
"""

from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
N_SPLITS = 5
SEEDS = [42, 10, 2026]
TOP_N_FEATURES = 150
USE_LOG_TRANSFORM = True
DROP_COLS = ["layout_id_target_enc", "layout_cluster_target_enc"]

LAG_COLS = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "battery_mean"]
ROLLING_COLS = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "robot_active", "battery_mean"]
ROLLING_WINDOWS = [3, 4, 5]
SCENARIO_STAT_COLS = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "robot_active", "battery_mean"]


def merge_layout_info(
    train: pd.DataFrame,
    test: pd.DataFrame,
    layout_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    li = layout_info.copy()

    if {"floor_area_sqm", "robot_total"}.issubset(li.columns):
        li["floor_area_per_robot"] = li["floor_area_sqm"] / (li["robot_total"] + 1)
    if {"intersection_count", "floor_area_sqm"}.issubset(li.columns):
        li["intersection_density"] = li["intersection_count"] / (li["floor_area_sqm"] + 1)
    if {"layout_compactness", "aisle_width_avg"}.issubset(li.columns):
        li["aisle_compactness"] = li["layout_compactness"] / (li["aisle_width_avg"] + 1e-3)
    if {"one_way_ratio", "intersection_count"}.issubset(li.columns):
        li["route_constraint"] = li["one_way_ratio"] * li["intersection_count"]
    if {"robot_total", "charger_count"}.issubset(li.columns):
        li["charger_ratio"] = li["robot_total"] / (li["charger_count"] + 1)
    if {"pack_station_count", "floor_area_sqm"}.issubset(li.columns):
        li["pack_station_per_area"] = li["pack_station_count"] / (li["floor_area_sqm"] + 1)

    if {"aisle_count", "robot_total"}.issubset(li.columns):
        li["aisle_per_robot"] = li["aisle_count"] / (li["robot_total"] + 1)
    if {"pack_station_count", "robot_total"}.issubset(li.columns):
        li["station_per_robot"] = li["pack_station_count"] / (li["robot_total"] + 1)
    if {"robot_total", "floor_area_sqm", "aisle_count"}.issubset(li.columns):
        li["trip_intensity"] = li["robot_total"] / (li["floor_area_sqm"] * (li["aisle_count"] + 1) + 1)

    if "layout_type" in li.columns:
        li["layout_type_enc"] = li["layout_type"].astype("category").cat.codes
        li_merged = li.drop(columns=["layout_type"])
    else:
        li_merged = li

    train = train.merge(li_merged, on="layout_id", how="left")
    test = test.merge(li_merged, on="layout_id", how="left")
    return train, test, li


def add_layout_pca(
    train: pd.DataFrame,
    test: pd.DataFrame,
    layout_info_enriched: pd.DataFrame,
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = [c for c in layout_info_enriched.columns if c != "layout_id" and pd.api.types.is_numeric_dtype(layout_info_enriched[c])]
    if not num_cols:
        return train, test

    x = layout_info_enriched[num_cols].fillna(0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_comp = min(n_components, x_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    z = pca.fit_transform(x_scaled)

    pca_cols = [
        "layout_pca_capacity",
        "layout_pca_complexity",
        "layout_pca_efficiency",
    ][:n_comp]
    pca_df = pd.DataFrame(z, columns=pca_cols, index=layout_info_enriched.index)
    pca_df.insert(0, "layout_id", layout_info_enriched["layout_id"].values)

    train = train.merge(pca_df, on="layout_id", how="left")
    test = test.merge(pca_df, on="layout_id", how="left")
    return train, test


def _get_sort_keys(df: pd.DataFrame) -> list[str]:
    keys = ["scenario_id"]
    for cand in ["time_idx", "time_step", "timeslot", "timestamp", "ID"]:
        if cand in df.columns:
            keys.append(cand)
            break
    return keys


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario_id" not in df.columns:
        return df
    work = df.sort_values(_get_sort_keys(df)).copy()
    work["time_idx"] = work.groupby("scenario_id").cumcount()

    if "order_inflow_15m" in work.columns:
        work["cumulative_inflow"] = work.groupby("scenario_id")["order_inflow_15m"].cumsum()
        work["cumulative_inflow_lag1"] = work.groupby("scenario_id")["cumulative_inflow"].shift(1).fillna(0)

    if "congestion_score" in work.columns:
        work["congestion_velocity"] = (
            work["congestion_score"] - work.groupby("scenario_id")["congestion_score"].shift(1)
        ).fillna(0)

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


def add_rolling_features(df: pd.DataFrame, base_cols: list[str], windows: list[int]) -> pd.DataFrame:
    if "scenario_id" not in df.columns:
        return df

    work = df.sort_values(_get_sort_keys(df)).copy()
    for col in base_cols:
        if col not in work.columns:
            continue
        grp = work.groupby("scenario_id")[col]
        shifted = grp.shift(1)
        for w in windows:
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


def add_ratio_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    new: dict[str, pd.Series] = {}

    if {"robot_active", "floor_area_sqm"}.issubset(df.columns):
        new["robot_active_per_area"] = df["robot_active"] / (df["floor_area_sqm"] + 1)

    if {"order_inflow_15m", "aisle_count"}.issubset(df.columns):
        new["inflow_per_aisle"] = df["order_inflow_15m"] / (df["aisle_count"] + 1)

    if {"congestion_score", "scenario_id"}.issubset(df.columns):
        scen_mean = df.groupby("scenario_id")["congestion_score"].transform("mean")
        new["congestion_vs_scenario_mean"] = df["congestion_score"] / (scen_mean + 1e-6)

    if {"order_inflow_15m", "robot_active"}.issubset(df.columns):
        new["ops_pressure_idx"] = df["order_inflow_15m"] / (df["robot_active"] + 1)

    if {"staff_on_floor", "robot_active"}.issubset(df.columns):
        new["human_robot_density"] = df["staff_on_floor"] / (df["robot_active"] + 1)

    if {"congestion_score", "worker_avg_tenure_months"}.issubset(df.columns):
        new["congestion_tenure_weighted"] = df["congestion_score"] / (df["worker_avg_tenure_months"] + 1)

    if {"network_latency_ms", "wifi_signal_db"}.issubset(df.columns):
        new["network_instability"] = df["network_latency_ms"] - df["wifi_signal_db"]
        new["network_bottleneck"] = df["network_latency_ms"] / (df["wifi_signal_db"].abs() + 1)

    if {"ambient_noise_db", "floor_vibration_idx"}.issubset(df.columns):
        new["environment_stress"] = df["ambient_noise_db"] + df["floor_vibration_idx"]
        new["sensor_noise_stress"] = df["floor_vibration_idx"] * np.log1p(df["ambient_noise_db"].clip(lower=0))

    if {"congestion_score", "sensor_noise_stress"}.issubset(df.columns):
        scaled = np.exp(np.clip(df["congestion_score"], -200, 200) / 100.0)
        new["congestion_noise_explosive"] = scaled * df["sensor_noise_stress"]

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def add_physical_mechanics_features(df: pd.DataFrame) -> pd.DataFrame:
    new: dict[str, pd.Series] = {}

    if {"order_inflow_15m", "pack_station_count", "battery_mean"}.issubset(df.columns):
        battery_pct = df["battery_mean"].where(df["battery_mean"] > 1, df["battery_mean"] * 100)
        new["saturation_index"] = df["order_inflow_15m"] / ((df["pack_station_count"] + 1) * (battery_pct / 100 + 1e-6))

    if "pack_utilization" in df.columns:
        util = df["pack_utilization"].clip(upper=0.999)
        new["explosion_factor"] = util / (1.01 - util)
    elif {"order_inflow_15m", "pack_station_count"}.issubset(df.columns):
        util_proxy = (df["order_inflow_15m"] / (df["pack_station_count"] + 1))
        util_proxy = util_proxy / (util_proxy + 20)
        util_proxy = util_proxy.clip(upper=0.999)
        new["explosion_factor"] = util_proxy / (1.01 - util_proxy)

    if "total_aisle_length" in df.columns:
        aisle_len = df["total_aisle_length"]
    elif {"aisle_count", "aisle_width_avg"}.issubset(df.columns):
        aisle_len = df["aisle_count"] * df["aisle_width_avg"]
    else:
        aisle_len = None

    robot_col = "robot_count" if "robot_count" in df.columns else ("robot_active" if "robot_active" in df.columns else None)
    if robot_col and "congestion_score" in df.columns and aisle_len is not None:
        new["density_stress"] = (df[robot_col] * df["congestion_score"]) / (aisle_len + 0.1)

    distance_col = None
    for c in ["avg_trip_distance", "mean_trip_distance", "trip_distance_mean", "route_constraint"]:
        if c in df.columns:
            distance_col = c
            break
    if distance_col and "order_inflow_15m" in df.columns and robot_col:
        new["workload_intensity"] = (df["order_inflow_15m"] * df[distance_col]) / (df[robot_col] + 1)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


def add_scenario_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, stat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def add_missing_flags(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_cols = pd.Index(train_df.columns[train_df.isna().mean() > 0]).unique()
    train_flags: dict[str, pd.Series] = {}
    test_flags: dict[str, pd.Series] = {}

    for col in missing_cols:
        if col == TARGET:
            continue
        flag_col = f"{col}_is_missing"

        tcol = train_df[col]
        if isinstance(tcol, pd.DataFrame):
            tcol = tcol.iloc[:, 0]
        train_flags[flag_col] = tcol.isna().astype("int8")

        if col in test_df.columns:
            scol = test_df[col]
            if isinstance(scol, pd.DataFrame):
                scol = scol.iloc[:, 0]
            test_flags[flag_col] = scol.isna().astype("int8")

    if train_flags:
        train_df = pd.concat([train_df, pd.DataFrame(train_flags, index=train_df.index)], axis=1)
    if test_flags:
        test_df = pd.concat([test_df, pd.DataFrame(test_flags, index=test_df.index)], axis=1)

    return train_df, test_df


def impute_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = train_df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    if "scenario_id" in train_df.columns:
        train_df[num_cols] = train_df.groupby("scenario_id")[num_cols].transform(lambda g: g.ffill().bfill())
    if "scenario_id" in test_df.columns:
        avail = [c for c in num_cols if c in test_df.columns]
        test_df[avail] = test_df.groupby("scenario_id")[avail].transform(lambda g: g.ffill().bfill())

    med = train_df[num_cols].median()
    train_df[num_cols] = train_df[num_cols].fillna(med)
    test_df[num_cols] = test_df[num_cols].fillna(med)
    return train_df, test_df


def build_lgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=seed,
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
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    if "layout_id" in train.columns:
        groups = train["layout_id"]
    elif "scenario_id" in train.columns:
        groups = train["scenario_id"]
    else:
        groups = np.arange(len(train))

    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_preds = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)
    imp_sum = pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train[TARGET], groups=groups), start=1):
        print(f"--- Fold {fold} (seed={seed}) ---")

        x_tr = train.loc[tr_idx, feature_cols]
        x_val = train.loc[val_idx, feature_cols]
        y_tr = _transform_target(train.loc[tr_idx, TARGET])
        y_val = _transform_target(train.loc[val_idx, TARGET])

        model = build_lgbm(seed)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)],
        )

        oof_preds[val_idx] = _inverse_target(model.predict(x_val))
        test_preds += _inverse_target(model.predict(test[feature_cols])) / N_SPLITS
        imp_sum += pd.Series(model.feature_importances_, index=feature_cols)

    return oof_preds, test_preds, imp_sum / N_SPLITS


def select_top_features(feature_cols: list[str], importance: pd.Series, top_n: int = TOP_N_FEATURES) -> list[str]:
    ranked = importance.sort_values(ascending=False)
    selected = ranked.head(top_n).index.tolist()
    if len(selected) < 50:
        selected = ranked.head(50).index.tolist()

    dropped = [c for c in feature_cols if c not in selected]
    print(f"\n[v10-prune] feature diet: {len(feature_cols)} -> {len(selected)}")
    if dropped:
        print(f"[v10-prune] dropped sample: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
    return selected


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    layout_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train, test, li_enriched = merge_layout_info(train, test, layout_info)
    train, test = add_layout_pca(train, test, li_enriched, n_components=3)

    for name, df in [("train", train), ("test", test)]:
        df = add_time_features(df)
        df = add_ratio_and_interaction_features(df)
        df = add_physical_mechanics_features(df)
        df = add_lag_features(df, LAG_COLS)
        df = add_rolling_features(df, ROLLING_COLS, ROLLING_WINDOWS)
        if name == "train":
            train = df
        else:
            test = df

    train, test = add_scenario_stats(train, test, SCENARIO_STAT_COLS)
    train, test = add_missing_flags(train, test)

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    feature_cols = [c for c in feature_cols if c in test.columns and c not in DROP_COLS]

    train, test = impute_features(train, test, feature_cols)
    feature_cols = [c for c in feature_cols if c in test.columns]
    feature_cols = train[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    return train, test, feature_cols


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"
    layout_info_path = project_root / "data" / "meta" / "layout_info.csv"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_root / "data" / "submission" / f"submission_v10_{ts}.csv"
    fi_path = project_root / "reports" / "eda" / "feature_importance_v10.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    layout_info = pd.read_csv(layout_info_path)

    print(f"학습 데이터: {train.shape}  테스트 데이터: {test.shape}  레이아웃: {layout_info.shape}")
    print(f"log transform: {USE_LOG_TRANSFORM} / seeds: {SEEDS}")

    train, test, feature_cols = build_features(train, test, layout_info)
    print(f"최종 피처 수: {len(feature_cols)}")

    imp_seed_sum = pd.Series(np.zeros(len(feature_cols), dtype=float), index=feature_cols)
    for seed in SEEDS:
        print(f"\n[Stage1] seed={seed}")
        _, _, imp_seed = run_cv(train, test, feature_cols, seed)
        imp_seed_sum += imp_seed
    imp_avg = imp_seed_sum / len(SEEDS)

    top_features = select_top_features(feature_cols, imp_avg, top_n=TOP_N_FEATURES)
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    imp_avg.sort_values(ascending=False).to_csv(fi_path, header=["importance"])
    print(f"feature importance 저장 완료: {fi_path}")

    oof_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []
    for seed in SEEDS:
        print(f"\n[Stage2] seed={seed}")
        oof_seed, test_seed, _ = run_cv(train, test, top_features, seed)
        oof_list.append(oof_seed)
        test_list.append(test_seed)

    oof_preds = np.mean(np.vstack(oof_list), axis=0)
    test_preds = np.mean(np.vstack(test_list), axis=0)

    oof_mae = mean_absolute_error(train[TARGET], oof_preds)
    print(f"\nOOF MAE: {oof_mae:.6f}")

    submission = pd.DataFrame({"ID": test["ID"], TARGET: test_preds})
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"submission 저장 완료: {submission_path}")


if __name__ == "__main__":
    main()
