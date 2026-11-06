from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import imblearn
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Class"
NORMAL, FRAUD = 0, 1

FALSE_NEGATIVE_COST = 100.0
FALSE_POSITIVE_COST = 2.0
DEFAULT_THRESHOLD = 0.5
MODEL_VERSION = "1.0.0"

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

COLORS = {NORMAL: "#3b6ea5", FRAUD: "#c0392b"}
LABELS = {NORMAL: "Normal", FRAUD: "Fraud"}
PLOT_DPI = 150

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update(
    {
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "savefig.bbox": "tight",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, threshold-tune and export the fraud detection model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("creditcard.csv"),
        help="Path to creditcard.csv (default: ./creditcard.csv).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for model/scaler/metadata artifacts (default: ./artifacts).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for diagnostic figures (default: ./plots).",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(
            f"Dataset not found: {path.resolve()}\n"
            "Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and pass "
            "its path with --data."
        )

    print(f"Loading dataset from {path.resolve()} ...")
    df = pd.read_csv(path)

    required = {TARGET, "Amount", "Time"} | {f"V{i}" for i in range(1, 29)}
    missing = sorted(required.difference(df.columns))
    if missing:
        sys.exit(f"Dataset is missing required column(s): {', '.join(missing)}")

    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def run_eda(df: pd.DataFrame, plots_dir: Path) -> None:
    print("\n" + "=" * 74)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 74)

    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:,.1f} MB")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\nMissing values (non-zero columns only):")
    print(missing.to_string() if not missing.empty else "  none")

    n_duplicates = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {n_duplicates:,} ({n_duplicates / len(df):.3%}) - retained")

    counts = df[TARGET].value_counts().sort_index()
    print("\nClass distribution:")
    for label, count in counts.items():
        print(f"  {LABELS.get(label, label):<8} {count:>8,}  ({count / len(df):.4%})")
    print(f"  Imbalance ratio: 1 fraud per "
          f"{counts[NORMAL] / max(counts[FRAUD], 1):,.0f} normal transactions")

    print("\nAmount statistics by class (USD):")
    print(
        df.groupby(TARGET)["Amount"]
        .agg(["count", "mean", "median", "max"])
        .rename(index=LABELS)
        .round(2)
        .to_string()
    )

    plot_class_distribution(df, plots_dir)
    plot_amount_distribution(df, plots_dir)
    plot_correlation_matrix(df, plots_dir)


def plot_class_distribution(df: pd.DataFrame, plots_dir: Path) -> None:
    counts = df[TARGET].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bars = ax.bar(
        [LABELS[NORMAL], LABELS[FRAUD]],
        counts.values,
        color=[COLORS[NORMAL], COLORS[FRAUD]],
        width=0.5,
    )
    ax.set_yscale("log")
    ax.set_ylim(1, counts.max() * 8)
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Raw Class Distribution — Extreme Imbalance")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.25,
            f"{value:,}\n({value / len(df):.3%})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.text(
        0.99,
        0.02,
        f"{len(df):,} transactions | {counts[FRAUD]:,} frauds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, plots_dir, "class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for label in (NORMAL, FRAUD):
        subset = df.loc[df[TARGET] == label, "Amount"]
        label_text = f"{LABELS[label]} (n={len(subset):,})"
        sns.kdeplot(
            x=subset,
            ax=axes[0],
            color=COLORS[label],
            fill=True,
            alpha=0.35,
            linewidth=1.8,
            clip=(0, 2500),
            common_norm=False,
            label=label_text,
        )
        sns.kdeplot(
            x=np.log1p(subset),
            ax=axes[1],
            color=COLORS[label],
            fill=True,
            alpha=0.35,
            linewidth=1.8,
            common_norm=False,
            label=label_text,
        )
        axes[0].axvline(subset.median(), color=COLORS[label], linestyle="--", linewidth=1)
        axes[1].axvline(np.log1p(subset.median()), color=COLORS[label], linestyle="--", linewidth=1)

    axes[0].set_xlim(0, 1500)
    axes[0].set_xlabel("Transaction amount (USD)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount KDE — Normal vs Fraud")
    axes[0].legend(frameon=True)

    axes[1].set_xlabel("log1p(transaction amount)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Amount KDE on log1p Scale")

    fig.suptitle("Transaction Amount Distribution by Class", fontsize=14, fontweight="bold")
    save_figure(fig, plots_dir, "amount_distribution.png")


def plot_correlation_matrix(df: pd.DataFrame, plots_dir: Path) -> None:
    corr = df.corr(numeric_only=True)
    order = corr[TARGET].abs().sort_values(ascending=False).index
    corr = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (ordered by |correlation| with Class)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    save_figure(fig, plots_dir, "correlation_matrix.png")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\n" + "=" * 74)
    print("TRAIN / TEST SPLIT (before scaling and resampling)")
    print("=" * 74)
    print(f"Train: {len(X_train):,} rows | frauds={int(y_train.sum()):,} "
          f"({y_train.mean():.4%})")
    print(f"Test : {len(X_test):,} rows | frauds={int(y_test.sum()):,} "
          f"({y_test.mean():.4%})")
    return X_train, X_test, y_train, y_test


def fit_scaler(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    columns = ["Time", "Amount"]
    scaler = StandardScaler().fit(X_train[columns])

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        scaled = frame.copy()
        scaled[columns] = scaler.transform(frame[columns])
        return scaled.rename(columns={"Time": "scaled_time", "Amount": "scaled_amount"})

    X_train_scaled = transform(X_train)
    X_test_scaled = transform(X_test)

    print("\nScaler fitted on X_train[['Time', 'Amount']] only.")
    print(f"Model features ({X_train_scaled.shape[1]}): "
          f"{X_train_scaled.columns[0]}, V1...V28, {X_train_scaled.columns[-1]}")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    minority = int(y_train.sum())
    k_neighbors = max(1, min(5, minority - 1))

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("\n" + "=" * 74)
    print("SMOTE (training partition only; test set untouched)")
    print("=" * 74)
    print(f"Before: {len(y_train):,} rows | frauds={minority:,} ({y_train.mean():.4%})")
    print(f"After : {len(y_resampled):,} rows | frauds={int(y_resampled.sum()):,} "
          f"({y_resampled.mean():.4%}) | k_neighbors={k_neighbors}")
    return X_resampled, y_resampled


def metrics_at_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NORMAL, FRAUD]).ravel()
    cost = fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "alerts": int(tp + fp),
        "expected_cost_usd": round(float(cost), 2),
    }


def build_threshold_table(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [metrics_at_threshold(y_true, y_score, t) for t in THRESHOLD_GRID]
    )


def select_optimal_threshold(table: pd.DataFrame) -> dict:
    best = table.sort_values(
        ["expected_cost_usd", "alerts", "threshold"],
        ascending=[True, True, False],
    ).iloc[0].to_dict()

    integer_fields = {"tp", "fp", "fn", "tn", "alerts"}
    return {
        key: int(value) if key in integer_fields else float(value)
        for key, value in best.items()
    }


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> dict:
    started = time.perf_counter()
    y_score = model.predict_proba(X_test)[:, FRAUD]
    inference_seconds = time.perf_counter() - started

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = float(auc(recall_curve, precision_curve))
    average_precision = float(average_precision_score(y_test, y_score))

    table = build_threshold_table(y_test, y_score)
    optimal = select_optimal_threshold(table)
    default = metrics_at_threshold(y_test, y_score, DEFAULT_THRESHOLD)

    print("\n" + "-" * 74)
    print(f"{model_name} — evaluation on the un-sampled test set")
    print("-" * 74)
    print(f"PR-AUC: {pr_auc:.4f} | Average precision: {average_precision:.4f}")
    print(f"Default threshold 0.50 : precision={default['precision']:.4f} "
          f"recall={default['recall']:.4f} f1={default['f1']:.4f} "
          f"cost=${default['expected_cost_usd']:,.0f}")
    print(f"Optimal threshold {optimal['threshold']:.2f} : "
          f"precision={optimal['precision']:.4f} recall={optimal['recall']:.4f} "
          f"f1={optimal['f1']:.4f} cost=${optimal['expected_cost_usd']:,.0f}")
    print(f"Hold-out inference over {len(X_test):,} rows: {inference_seconds * 1000:,.1f} ms")
    print("\nClassification report at the cost-optimal threshold:")
    optimal_pred = (y_score >= optimal["threshold"]).astype(int)
    print(classification_report(
        y_test, optimal_pred, target_names=["Normal", "Fraud"],
        digits=4, zero_division=0,
    ))

    return {
        "model_name": model_name,
        "y_score": y_score,
        "pr_curve": (recall_curve, precision_curve),
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "default": default,
        "optimal": optimal,
        "threshold_table": table,
    }


def plot_confusion_matrices(rf_result: dict, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    panels = (
        ("Default threshold = 0.50", rf_result["default"]),
        (f"Cost-optimal threshold = {rf_result['optimal']['threshold']:.2f}",
         rf_result["optimal"]),
    )
    for ax, (title, metrics) in zip(axes, panels):
        cm = np.array(
            [
                [int(metrics["tn"]), int(metrics["fp"])],
                [int(metrics["fn"]), int(metrics["tp"])],
            ]
        )
        shares = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        annot = np.array(
            [[f"{value:,}\n({share:.2f}%)" for value, share in zip(row, row_shares)]
             for row, row_shares in zip(cm, shares)]
        )
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=[LABELS[NORMAL], LABELS[FRAUD]],
            yticklabels=[LABELS[NORMAL], LABELS[FRAUD]],
            linewidths=0.8,
            linecolor="white",
            annot_kws={"fontsize": 12},
            ax=ax,
        )
        ax.set_title(
            f"{title}\ncost = ${metrics['expected_cost_usd']:,.0f} "
            f"(FN={metrics['fn']}, FP={metrics['fp']})"
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")

    fig.suptitle(
        "Random Forest — Confusion Matrices at Default vs Cost-Optimal Threshold",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, plots_dir, "confusion_matrices.png")


def plot_pr_curves(results: dict[str, dict], y_test: pd.Series, plots_dir: Path) -> None:
    baseline = float(y_test.mean())
    palette = ["#2c7fb8", "#31a354"]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for index, (name, result) in enumerate(results.items()):
        recall, precision = result["pr_curve"]
        color = palette[index % len(palette)]
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{name} (PR-AUC = {result['pr_auc']:.4f})",
        )
        ax.scatter(
            [result["optimal"]["recall"]],
            [result["optimal"]["precision"]],
            color=color,
            edgecolor="black",
            zorder=5,
            s=55,
        )
        ax.annotate(
            f"threshold {result['optimal']['threshold']:.2f}",
            (result["optimal"]["recall"], result["optimal"]["precision"]),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=8,
            color=color,
        )

    ax.axhline(
        baseline,
        color="grey",
        linestyle="--",
        linewidth=1.2,
        label=f"No-skill baseline = {baseline:.4f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (alerts that are fraud)")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.3)
    save_figure(fig, plots_dir, "pr_curves.png")


def plot_cost_curve(results: dict[str, dict], plots_dir: Path) -> None:
    palette = ["#2c7fb8", "#31a354"]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for index, (name, result) in enumerate(results.items()):
        table = result["threshold_table"]
        color = palette[index % len(palette)]
        label = f"{name} (min ${result['optimal']['expected_cost_usd']:,.0f} " \
                f"@ {result['optimal']['threshold']:.2f})"
        ax.plot(table["threshold"], table["expected_cost_usd"],
                color=color, linewidth=2, label=label)
        ax.scatter(
            [result["optimal"]["threshold"]],
            [result["optimal"]["expected_cost_usd"]],
            color=color,
            edgecolor="black",
            s=70,
            zorder=5,
        )
        ax.annotate(
            f"${result['optimal']['expected_cost_usd']:,.0f}",
            (result["optimal"]["threshold"], result["optimal"]["expected_cost_usd"]),
            textcoords="offset points",
            xytext=(8, 10),
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.axvline(DEFAULT_THRESHOLD, color="#777777", linestyle=":", linewidth=1.2,
               label="Default threshold 0.50")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 0.99)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (USD, log scale)")
    ax.set_title(
        f"Cost Curve — FN=${FALSE_NEGATIVE_COST:,.0f}, FP=${FALSE_POSITIVE_COST:,.0f}"
    )
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, plots_dir, "cost_curve.png")


def plot_feature_importance(
    model: RandomForestClassifier, feature_names: list[str], plots_dir: Path
) -> None:
    importance = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(
        x=importance.values,
        y=importance.index,
        hue=importance.index,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for position, value in enumerate(importance.values):
        ax.text(value + importance.max() * 0.01, position, f"{value:.4f}",
                va="center", fontsize=9)
    ax.set_xlim(0, importance.max() * 1.18)
    ax.set_xlabel("Mean decrease in impurity (Gini importance)")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Feature Importances — Random Forest")
    save_figure(fig, plots_dir, "feature_importance.png")


def export_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_names: list[str],
    metadata: dict,
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "fraud_random_forest_model.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for path in (model_path, scaler_path, metadata_path):
        print(f"  exported -> {path} ({path.stat().st_size / 1024 ** 2:,.2f} MB)")


def save_figure(fig: plt.Figure, plots_dir: Path, filename: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plots_dir = args.plots_dir
    artifacts_dir = args.artifacts_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    run_eda(df, plots_dir)

    X_train_raw, X_test_raw, y_train, y_test = split_data(df)
    X_train, X_test, scaler = fit_scaler(X_train_raw, X_test_raw)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        started = time.perf_counter()
        model.fit(X_train_smote, y_train_smote)
        print(f"  fitted in {time.perf_counter() - started:,.1f}s")
        results[name] = evaluate_model(model, X_test, y_test, name)

    rf_result = results["Random Forest"]
    lr_result = results["Logistic Regression"]

    print("\n" + "=" * 74)
    print("BENCHMARK SUMMARY — un-sampled test set")
    print("=" * 74)
    summary_rows = []
    for name, result in results.items():
        for point in ("default", "optimal"):
            metrics = result[point]
            summary_rows.append(
                {
                    "model": name,
                    "threshold": metrics["threshold"],
                    "pr_auc": round(result["pr_auc"], 4),
                    "accuracy": round(metrics["accuracy"], 4),
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1"], 4),
                    "cost_usd": metrics["expected_cost_usd"],
                }
            )
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    approve_all_cost = int(y_test.sum()) * FALSE_NEGATIVE_COST
    print(f"\nCost of approving every transaction: ${approve_all_cost:,.0f}")
    print(f"Random Forest @ optimal threshold  : "
          f"${rf_result['optimal']['expected_cost_usd']:,.0f} "
          f"(saves ${approve_all_cost - rf_result['optimal']['expected_cost_usd']:,.0f})")

    plot_confusion_matrices(rf_result, plots_dir)
    plot_pr_curves(results, y_test, plots_dir)
    plot_cost_curve(results, plots_dir)
    plot_feature_importance(models["Random Forest"], list(X_train.columns), plots_dir)

    feature_names = list(X_train.columns)
    metadata = {
        "model_name": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "optimal_threshold": round(float(rf_result["optimal"]["threshold"]), 4),
        "default_threshold": DEFAULT_THRESHOLD,
        "feature_names": feature_names,
        "scaler_features": ["Time", "Amount"],
        "cost_matrix": {
            "false_negative_cost_usd": FALSE_NEGATIVE_COST,
            "false_positive_cost_usd": FALSE_POSITIVE_COST,
        },
        "random_forest": {
            "pr_auc": round(float(rf_result["pr_auc"]), 6),
            "average_precision": round(float(rf_result["average_precision"]), 6),
            "metrics_at_default_threshold": rf_result["default"],
            "metrics_at_optimal_threshold": rf_result["optimal"],
        },
        "logistic_regression": {
            "pr_auc": round(float(lr_result["pr_auc"]), 6),
            "average_precision": round(float(lr_result["average_precision"]), 6),
            "metrics_at_default_threshold": lr_result["default"],
            "metrics_at_optimal_threshold": lr_result["optimal"],
        },
        "dataset": {
            "rows": int(len(df)),
            "frauds": int(df[TARGET].sum()),
            "fraud_rate": round(float(df[TARGET].mean()), 6),
            "train_rows": int(len(y_train)),
            "train_frauds": int(y_train.sum()),
            "test_rows": int(len(y_test)),
            "test_frauds": int(y_test.sum()),
            "train_rows_after_smote": int(len(y_train_smote)),
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
        },
    }

    print("\nExporting artifacts ...")
    export_artifacts(
        models["Random Forest"], scaler, feature_names, metadata, artifacts_dir
    )

    print(f"\nDone. Artifacts: {artifacts_dir.resolve()} | Plots: {plots_dir.resolve()}")
    print(f"Deploy with: uvicorn app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
