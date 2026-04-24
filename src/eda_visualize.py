from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TARGET = "avg_delay_minutes_next_30m"
ID_COLS = ["ID", "layout_id", "scenario_id"]
MAX_HEATMAP_FEATURES = 35
TOP_CORR_FEATURES = 20
SAMPLE_FOR_PAIRPLOT = 5000


def save_missing_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    missing_ratio = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_ratio = missing_ratio[missing_ratio > 0].head(30)

    if missing_ratio.empty:
        print("결측치가 없어 missing plot을 생략합니다.")
        return

    plt.figure(figsize=(11, 8))
    sns.barplot(x=missing_ratio.values, y=missing_ratio.index, orient="h")
    plt.title("Top Missing Ratio Features (%)")
    plt.xlabel("Missing Ratio (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "01_missing_ratio_top30.png", dpi=180)
    plt.close()


def save_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET], bins=60, kde=True)
    plt.title("Target Distribution")
    plt.xlabel(TARGET)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "02_target_distribution.png", dpi=180)
    plt.close()


def save_target_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET not in numeric_cols:
        print("타깃이 숫자형이 아니어서 상관도 그래프를 생략합니다.")
        return

    corr = df[numeric_cols].corr(numeric_only=True)[TARGET].dropna()
    corr = corr.drop(labels=[TARGET], errors="ignore")
    top = corr.reindex(corr.abs().sort_values(ascending=False).head(TOP_CORR_FEATURES).index)

    plt.figure(figsize=(11, 8))
    sns.barplot(x=top.values, y=top.index, orient="h")
    plt.title(f"Top {TOP_CORR_FEATURES} Correlations with Target")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / "03_target_correlation_top20.png", dpi=180)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET not in numeric_cols:
        print("타깃이 숫자형이 아니어서 히트맵을 생략합니다.")
        return

    corr_to_target = df[numeric_cols].corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
    selected = corr_to_target.head(MAX_HEATMAP_FEATURES).index.tolist()
    heat_df = df[selected].corr(numeric_only=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        heat_df,
        cmap="RdBu_r",
        center=0,
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Correlation Heatmap (Top {MAX_HEATMAP_FEATURES} by |corr with target|)")
    plt.tight_layout()
    plt.savefig(out_dir / "04_correlation_heatmap_top35.png", dpi=180)
    plt.close()


def save_pairplot(df: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET not in numeric_cols:
        return

    corr_to_target = df[numeric_cols].corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
    pair_cols = [c for c in corr_to_target.index if c != TARGET][:5] + [TARGET]

    sampled = df[pair_cols].dropna()
    if len(sampled) > SAMPLE_FOR_PAIRPLOT:
        sampled = sampled.sample(SAMPLE_FOR_PAIRPLOT, random_state=42)

    g = sns.pairplot(sampled, corner=True, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.4})
    g.fig.suptitle("Pairplot of Top Correlated Features", y=1.02)
    g.savefig(out_dir / "05_pairplot_top_features.png", dpi=160)
    plt.close("all")


def main() -> None:
    sns.set_theme(style="whitegrid")

    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "train.csv"
    out_dir = project_root / "reports" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    print(f"Loaded train data: {df.shape}")

    save_missing_ratio(df, out_dir)
    save_target_distribution(df, out_dir)
    save_target_correlation(df, out_dir)
    save_correlation_heatmap(df, out_dir)
    save_pairplot(df, out_dir)

    print(f"EDA 이미지 저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
