"""Improved baseline for smart warehouse delay prediction.

Applied improvements:
- Target log transform (train: log1p, inference: expm1)
- GroupKFold by scenario_id to reduce leakage
- Missing indicators + scenario-wise ffill/bfill + median fallback
- Interaction / lag / diff / scenario-stat features
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
N_SPLITS = 5


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"robot_charging", "robot_active"}.issubset(df.columns):
        df["charging_per_active"] = df["robot_charging"] / (df["robot_active"] + 1)

    if {"low_battery_ratio", "battery_mean"}.issubset(df.columns):
        df["low_battery_x_battery_mean"] = df["low_battery_ratio"] * df["battery_mean"]

    if {"order_inflow_15m", "congestion_score"}.issubset(df.columns):
        df["inflow_x_congestion"] = df["order_inflow_15m"] * df["congestion_score"]

    return df


def add_missing_flags(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def add_lag_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    if "scenario_id" not in df.columns:
        return df

    sort_keys = ["scenario_id"]
    for candidate in ["time_idx", "time_step", "timeslot", "timestamp", "ID"]:
        if candidate in df.columns:
            sort_keys.append(candidate)
            break

    work = df.sort_values(sort_keys).copy()

    for col in base_cols:
        if col not in work.columns:
            continue
        grp = work.groupby("scenario_id")[col]
        work[f"{col}_lag1"] = grp.shift(1)
        work[f"{col}_lag2"] = grp.shift(2)
        work[f"{col}_diff1"] = work[col] - work[f"{col}_lag1"]

    return work.sort_index()


def add_scenario_stats(train_df: pd.DataFrame, test_df: pd.DataFrame, stat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "scenario_id" not in train_df.columns:
        return train_df, test_df

    for col in stat_cols:
        if col not in train_df.columns:
            continue

        grp = train_df.groupby("scenario_id")[col].agg(["mean", "std"])
        mean_map = grp["mean"]
        std_map = grp["std"]

        train_df[f"{col}_scenario_mean"] = train_df["scenario_id"].map(mean_map)
        train_df[f"{col}_scenario_std"] = train_df["scenario_id"].map(std_map)

        test_df[f"{col}_scenario_mean"] = test_df["scenario_id"].map(mean_map)
        test_df[f"{col}_scenario_std"] = test_df["scenario_id"].map(std_map)

    return train_df, test_df


def impute_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = train_df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    if "scenario_id" in train_df.columns:
        train_df[numeric_cols] = train_df.groupby("scenario_id")[numeric_cols].transform(lambda g: g.ffill().bfill())
    if "scenario_id" in test_df.columns:
        available = [c for c in numeric_cols if c in test_df.columns]
        test_df[available] = test_df.groupby("scenario_id")[available].transform(lambda g: g.ffill().bfill())

    medians = train_df[numeric_cols].median()
    train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(medians)

    return train_df, test_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"
    submission_path = project_root / "data" / "submission" / "submission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"학습 데이터 크기: {train.shape}")
    print(f"테스트 데이터 크기: {test.shape}")

    train = add_interaction_features(train)
    test = add_interaction_features(test)

    lag_targets = ["order_inflow_15m", "congestion_score", "low_battery_ratio"]
    train = add_lag_features(train, lag_targets)
    test = add_lag_features(test, lag_targets)

    scenario_stat_targets = ["order_inflow_15m", "congestion_score", "low_battery_ratio", "robot_active"]
    train, test = add_scenario_stats(train, test, scenario_stat_targets)

    train, test = add_missing_flags(train, test)

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    train, test = impute_features(train, test, feature_cols)

    # 모델 입력은 숫자형 중심으로 구성해 안정적으로 학습시킨다.
    feature_cols = [c for c in feature_cols if c in test.columns]
    feature_cols = train[feature_cols].select_dtypes(include=["number", "bool"]).columns.tolist()

    print(f"최종 피처 수: {len(feature_cols)}")

    groups = train["scenario_id"] if "scenario_id" in train.columns else None
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_preds = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train[TARGET], groups=groups), start=1):
        print(f"--- Fold {fold} ---")

        X_tr = train.loc[tr_idx, feature_cols]
        X_val = train.loc[val_idx, feature_cols]

        y_tr = train.loc[tr_idx, TARGET].clip(lower=0)
        y_val = train.loc[val_idx, TARGET].clip(lower=0)
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)

        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=1500,
            learning_rate=0.03,
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

        model.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_val, y_val_log)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
        )

        val_pred = np.expm1(model.predict(X_val))
        tst_pred = np.expm1(model.predict(test[feature_cols]))

        oof_preds[val_idx] = np.clip(val_pred, a_min=0, a_max=None)
        test_preds += np.clip(tst_pred, a_min=0, a_max=None) / N_SPLITS

    oof_mae = mean_absolute_error(train[TARGET], oof_preds)
    print(f"OOF MAE: {oof_mae:.6f}")

    submission = pd.DataFrame({"ID": test["ID"], TARGET: test_preds})
    submission.to_csv(submission_path, index=False)
    print(f"submission 저장 완료: {submission_path}")


if __name__ == "__main__":
    main()
