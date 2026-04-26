"""Feature importance report generator (v3+).

v3부터 시각화는 학습 코드와 완전 분리.
baseline_lightgbm_v3.py 실행 후 생성된 feature_importance_v3.csv를 읽어
PNG 리포트를 생성합니다.

사용법:
    python src/report_importance.py
    python src/report_importance.py --csv reports/eda/feature_importance_v3.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LAYOUT_KEYS = [
    "floor_area_per_robot", "intersection_density", "aisle_compactness",
    "route_constraint", "charger_ratio", "pack_station_per_area",
    "charger_load", "inflow_per_station", "active_area_ratio",
    "layout_group", "layout_id_target_enc", "layout_type_enc",
    "battery_stress", "battery_risk",
]

V3_NEW_KEYS = ["time_idx", "cumulative_inflow", "cumulative_inflow_lag1"]


def load_importance(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)
    return df.iloc[:, 0].sort_values(ascending=False)


def plot_top_n(importance: pd.Series, out_path: Path, top_n: int = 40) -> None:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(12, max(6, top_n // 3)))
    top[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance Top {top_n} (fold-averaged gain)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path.name}")


def plot_layout_features(importance: pd.Series, out_path: Path) -> None:
    layout_imp = importance[importance.index.isin(LAYOUT_KEYS)].sort_values(ascending=False)
    if layout_imp.empty:
        print("[skip] layout 피처가 중요도 데이터에 없습니다.")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(layout_imp) // 2 + 1)))
    layout_imp[::-1].plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("Layout Feature Importance (fold-averaged gain)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path.name}")
    print(layout_imp.to_string())


def plot_v3_new_features(importance: pd.Series, out_path: Path) -> None:
    """v3 신규 피처(time_idx, cumulative_inflow 등) 중요도 확인."""
    new_imp = importance[importance.index.isin(V3_NEW_KEYS)].sort_values(ascending=False)
    if new_imp.empty:
        print("[skip] v3 신규 피처가 중요도 데이터에 없습니다.")
        return
    fig, ax = plt.subplots(figsize=(8, max(3, len(new_imp))))
    new_imp[::-1].plot(kind="barh", ax=ax, color="seagreen")
    ax.set_title("V3 New Feature Importance (time_idx, cumulative_inflow)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature importance visualization")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="feature_importance CSV 경로 (기본값: reports/eda/feature_importance_v3.csv)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "reports" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = out_dir / "feature_importance_v3.csv"

    if not csv_path.exists():
        print(f"[error] CSV 파일을 찾을 수 없습니다: {csv_path}")
        print("먼저 baseline_lightgbm_v3.py를 실행하세요.")
        return

    importance = load_importance(csv_path)
    print(f"피처 수: {len(importance)}")

    plot_top_n(importance, out_dir / "08_feature_importance_v3_top40.png")
    plot_layout_features(importance, out_dir / "09_layout_feature_importance_v3.png")
    plot_v3_new_features(importance, out_dir / "10_v3_new_features_importance.png")

    print("\n상위 10개 피처:")
    print(importance.head(10).to_string())


if __name__ == "__main__":
    main()
