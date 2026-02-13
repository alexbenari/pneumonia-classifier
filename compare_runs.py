import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix


STRATEGIES = [
    ("FC Only", "fc_only"),
    ("Full Fine-Tuning", "full_finetuning"),
    ("Partial Fine-Tuning", "partial_finetuning"),
]

ACC_METRIC_COLUMNS = [
    "Train Accuracy",
    "Val Accuracy",
    "Test Accuracy",
]

LOSS_METRIC_COLUMNS = [
    "Train Loss",
    "Val Loss",
    "Test Loss",
]

METRIC_COLUMNS = ACC_METRIC_COLUMNS + LOSS_METRIC_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two training runs.")
    parser.add_argument("--runA", help="First run name (directory under outputs/)")
    parser.add_argument("--runB", help="Second run name (directory under outputs/)")
    parser.add_argument("--outputs", default="outputs", help="Outputs directory (default: outputs)")
    return parser.parse_args()


def get_latest_runs(outputs_dir, count=2):
    runs = []
    if not os.path.isdir(outputs_dir):
        return []
    for entry in os.scandir(outputs_dir):
        if not entry.is_dir():
            continue
        summary_path = os.path.join(entry.path, "summary.txt")
        if os.path.isfile(summary_path):
            mtime = os.path.getmtime(summary_path)
        else:
            mtime = os.path.getmtime(entry.path)
        runs.append((entry.name, mtime))
    runs.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in runs[:count]]


def parse_run_timestamp(name):
    parts = name.split("_")
    if not parts:
        return None
    ts = parts[-1]
    if re.fullmatch(r"\d{8}-\d{6}", ts):
        return ts
    return None


def resolve_run_identifier(outputs_dir, value):
    if not value:
        return None
    candidate_dir = os.path.join(outputs_dir, value)
    if os.path.isdir(candidate_dir):
        return value

    prefix = f"{value}_"
    matches = []
    for entry in os.scandir(outputs_dir):
        if not entry.is_dir():
            continue
        if entry.name == value or entry.name.startswith(prefix):
            ts = parse_run_timestamp(entry.name)
            if ts is None:
                ts_key = ""
            else:
                ts_key = ts
            matches.append((entry.name, ts_key, os.path.getmtime(entry.path)))

    if not matches:
        raise ValueError(f"No runs found for config or run name '{value}'.")

    matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return matches[0][0]


def read_summary_from_txt(summary_path):
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    table_lines = []
    for line in lines:
        if line.strip() == "":
            break
        table_lines.append(line.rstrip("\n"))
    if not table_lines:
        raise ValueError(f"Summary table is empty in {summary_path}")

    header_tokens = [t.strip() for t in re.split(r"\s{2,}", table_lines[0].strip()) if t.strip()]
    rows = []
    for line in table_lines[1:]:
        if not line.strip():
            continue
        tokens = [t.strip() for t in re.split(r"\s{2,}", line.strip()) if t.strip()]
        if len(tokens) == len(header_tokens) + 1 and tokens[0].isdigit():
            tokens = tokens[1:]
        if len(tokens) != len(header_tokens):
            raise ValueError(
                f"Unable to parse summary row. Expected {len(header_tokens)} columns, "
                f"got {len(tokens)}. Line: {line}"
            )
        rows.append(tokens)

    df = pd.DataFrame(rows, columns=header_tokens)
    if "Strategy" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Strategy"})
    for col in df.columns:
        if col == "Strategy":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_summary_from_json(summary_path):
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Summary JSON has no metrics: {summary_path}")

    rows = []
    seen = set()
    for strategy, _ in STRATEGIES:
        if strategy in metrics:
            row = {"Strategy": strategy}
            row.update(metrics[strategy])
            rows.append(row)
            seen.add(strategy)
    for strategy, values in metrics.items():
        if strategy in seen:
            continue
        row = {"Strategy": strategy}
        row.update(values)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col == "Strategy":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_summary_table(run_dir):
    summary_json = os.path.join(run_dir, "summary.json")
    summary_txt = os.path.join(run_dir, "summary.txt")
    if os.path.isfile(summary_json):
        return read_summary_from_json(summary_json)
    if os.path.isfile(summary_txt):
        return read_summary_from_txt(summary_txt)
    raise FileNotFoundError(f"No summary.json or summary.txt found in {run_dir}")


def read_summary_json(run_dir):
    summary_json = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(summary_json):
        return None
    with open(summary_json, "r", encoding="utf-8") as f:
        return json.load(f)


def select_best_strategy(df):
    for col in ["Test Accuracy", "Test Loss"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in summary table.")
    best = df.sort_values(
        by=["Test Accuracy", "Test Loss"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return best["Strategy"]


def load_eval(run_dir, strategy_id):
    path = os.path.join(run_dir, f"eval_{strategy_id}.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=True)
    labels = data["labels"]
    preds = data["preds"]
    class_names = data["class_names"].tolist() if "class_names" in data else None
    return labels, preds, class_names


def load_classification_report(summary_json, run_dir, strategy_name, strategy_id):
    if summary_json:
        reports = summary_json.get("classification_reports", {})
        if strategy_name in reports:
            return reports[strategy_name]
    report_path = os.path.join(run_dir, f"classification_report_{strategy_id}.json")
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_class_metrics(report, class_name):
    if not report:
        return {}
    for key, value in report.items():
        if not isinstance(value, dict):
            continue
        if key.strip().lower() == class_name:
            return value
    return {}


def get_class_recall(report, class_name):
    metrics = get_class_metrics(report, class_name)
    value = metrics.get("recall", np.nan)
    return value if isinstance(value, (int, float, np.integer, np.floating)) else np.nan


def get_class_f1(report, class_name):
    metrics = get_class_metrics(report, class_name)
    value = metrics.get("f1-score", np.nan)
    return value if isinstance(value, (int, float, np.integer, np.floating)) else np.nan


def build_recall_f1_table(report_a, report_b):
    def get_accuracy(report):
        if not report:
            return np.nan
        value = report.get("accuracy", np.nan)
        return value if isinstance(value, (int, float, np.integer, np.floating)) else np.nan

    class_rows = ["normal", "pneumonia"]
    accuracy_a = get_accuracy(report_a)
    accuracy_b = get_accuracy(report_b)

    rows = []
    for class_name in class_rows:
        metrics_a = get_class_metrics(report_a, class_name)
        metrics_b = get_class_metrics(report_b, class_name)
        row = {"Category": class_name}
        row["Recall (A)"] = metrics_a.get("recall", np.nan)
        row["Recall (B)"] = metrics_b.get("recall", np.nan)
        row["Accuracy (A)"] = accuracy_a
        row["Accuracy (B)"] = accuracy_b
        row["F1 (A)"] = metrics_a.get("f1-score", np.nan)
        row["F1 (B)"] = metrics_b.get("f1-score", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Category")
    return df


def render_report_comparison(report_a, report_b, title):
    if report_a is None and report_b is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        render_placeholder(ax, "NA\n(no classification report data)")
        fig.suptitle(title, fontsize=12)
        return fig
    table = build_recall_f1_table(report_a, report_b)
    return render_table_page(table, title)


def render_placeholder(ax, label):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )


def plot_confusion(ax, labels, preds, class_names, title):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


def render_table_page(df, title):
    def format_value(value):
        if isinstance(value, (np.floating, float, np.integer, int)) and not isinstance(value, bool):
            return f"{float(value):.3f}"
        return value

    def format_display_df(input_df):
        display_df = input_df.copy()
        header_map = {
            "Train Accuracy (A)": "Train acc\n(A)",
            "Train Accuracy (B)": "Train acc\n(B)",
            "Val Accuracy (A)": "Val acc\n(A)",
            "Val Accuracy (B)": "Val acc\n(B)",
            "Test Accuracy (A)": "Test acc\n(A)",
            "Test Accuracy (B)": "Test acc\n(B)",
            "Test Recall (A)": "Test Recall\nPneu (A)",
            "Test Recall (B)": "Test Recall\nPneu (B)",
            "Test F1 (A)": "Test F1\nPneu (A)",
            "Test F1 (B)": "Test F1\nPneu (B)",
            "Train Loss (A)": "Train loss\n(A)",
            "Train Loss (B)": "Train loss\n(B)",
            "Val Loss (A)": "Val loss\n(A)",
            "Val Loss (B)": "Val loss\n(B)",
            "Test Loss (A)": "Test loss\n(A)",
            "Test Loss (B)": "Test loss\n(B)",
        }
        display_df = display_df.rename(columns=header_map)
        for col in display_df.columns:
            display_df[col] = display_df[col].map(format_value)
        return display_df

    display_df = format_display_df(df)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        rowLabels=display_df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    return fig


def render_strategy_comparison_page(main_df, loss_df, title):
    def format_value(value):
        if isinstance(value, (np.floating, float, np.integer, int)) and not isinstance(value, bool):
            return f"{float(value):.3f}"
        return value

    def format_display_df(input_df):
        display_df = input_df.copy()
        header_map = {
            "Train Accuracy (A)": "Train acc\n(A)",
            "Train Accuracy (B)": "Train acc\n(B)",
            "Val Accuracy (A)": "Val acc\n(A)",
            "Val Accuracy (B)": "Val acc\n(B)",
            "Test Accuracy (A)": "Test acc\n(A)",
            "Test Accuracy (B)": "Test acc\n(B)",
            "Test Recall (A)": "Test Recall\nPneu (A)",
            "Test Recall (B)": "Test Recall\nPneu (B)",
            "Test F1 (A)": "Test F1\nPneu (A)",
            "Test F1 (B)": "Test F1\nPneu (B)",
            "Train Loss (A)": "Train loss\n(A)",
            "Train Loss (B)": "Train loss\n(B)",
            "Val Loss (A)": "Val loss\n(A)",
            "Val Loss (B)": "Val loss\n(B)",
            "Test Loss (A)": "Test loss\n(A)",
            "Test Loss (B)": "Test loss\n(B)",
        }
        display_df = display_df.rename(columns=header_map)
        for col in display_df.columns:
            display_df[col] = display_df[col].map(format_value)
        return display_df

    main_display = format_display_df(main_df)
    loss_display = format_display_df(loss_df)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={"height_ratios": [2.2, 1.0]})
    fig.suptitle(title, fontsize=12)

    axes[0].axis("off")
    axes[0].set_title("Core Metrics", pad=8)
    table_main = axes[0].table(
        cellText=main_display.values,
        colLabels=main_display.columns,
        rowLabels=main_display.index,
        cellLoc="center",
        loc="center",
    )
    table_main.auto_set_font_size(False)
    table_main.set_fontsize(8)
    table_main.scale(1, 1.5)

    axes[1].axis("off")
    axes[1].set_title("Loss Metrics", pad=8)
    table_loss = axes[1].table(
        cellText=loss_display.values,
        colLabels=loss_display.columns,
        rowLabels=loss_display.index,
        cellLoc="center",
        loc="center",
    )
    table_loss.auto_set_font_size(False)
    table_loss.set_fontsize(8)
    table_loss.scale(1, 1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def build_comparison_table(df_a, df_b, report_map_a, report_map_b):
    df_a = df_a.set_index("Strategy")
    df_b = df_b.set_index("Strategy")
    rows = []
    for strategy, _ in STRATEGIES:
        row = {"Strategy": strategy}
        for metric in ACC_METRIC_COLUMNS:
            row[f"{metric} (A)"] = df_a.get(metric, pd.Series()).get(strategy, np.nan)
            row[f"{metric} (B)"] = df_b.get(metric, pd.Series()).get(strategy, np.nan)
        row["Test Recall (A)"] = get_class_recall(report_map_a.get(strategy), "pneumonia")
        row["Test Recall (B)"] = get_class_recall(report_map_b.get(strategy), "pneumonia")
        row["Test F1 (A)"] = get_class_f1(report_map_a.get(strategy), "pneumonia")
        row["Test F1 (B)"] = get_class_f1(report_map_b.get(strategy), "pneumonia")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Strategy")
    return df


def build_loss_comparison_table(df_a, df_b):
    df_a = df_a.set_index("Strategy")
    df_b = df_b.set_index("Strategy")
    rows = []
    for strategy, _ in STRATEGIES:
        row = {"Strategy": strategy}
        for metric in LOSS_METRIC_COLUMNS:
            row[f"{metric} (A)"] = df_a.get(metric, pd.Series()).get(strategy, np.nan)
            row[f"{metric} (B)"] = df_b.get(metric, pd.Series()).get(strategy, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Strategy")
    return df


def build_best_only_table(df_a, df_b, best_a, best_b):
    df_a = df_a.set_index("Strategy")
    df_b = df_b.set_index("Strategy")
    if best_a not in df_a.index or best_b not in df_b.index:
        raise ValueError("Best strategy not found in summary table.")
    row_a = df_a.loc[best_a]
    row_b = df_b.loc[best_b]
    col_a = "A (best)"
    col_b = "B (best)"
    rows = [{"Metric": "Strategy", col_a: best_a, col_b: best_b}]
    for metric in METRIC_COLUMNS:
        rows.append(
            {
                "Metric": metric,
                col_a: row_a.get(metric, np.nan),
                col_b: row_b.get(metric, np.nan),
            }
        )
    return pd.DataFrame(rows).set_index("Metric")


def render_cm_pair(eval_a, eval_b, strategy_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Confusion matrices - {strategy_name} (test set)", fontsize=12)

    if eval_a is None:
        render_placeholder(axes[0], "NA\n(no eval data)")
    else:
        labels, preds, class_names = eval_a
        class_names = class_names or ["Class 0", "Class 1"]
        plot_confusion(axes[0], labels, preds, class_names, "A")

    if eval_b is None:
        render_placeholder(axes[1], "NA\n(no eval data)")
    else:
        labels, preds, class_names = eval_b
        class_names = class_names or ["Class 0", "Class 1"]
        plot_confusion(axes[1], labels, preds, class_names, "B")

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    outputs_dir = args.outputs
    run_a = resolve_run_identifier(outputs_dir, args.runA)
    run_b = resolve_run_identifier(outputs_dir, args.runB)

    if not run_a or not run_b:
        latest = get_latest_runs(outputs_dir, count=2)
        if len(latest) < 2:
            raise ValueError("Not enough runs found to compare.")
        run_a, run_b = latest[0], latest[1]

    run_dir_a = os.path.join(outputs_dir, run_a)
    run_dir_b = os.path.join(outputs_dir, run_b)

    summary_a = read_summary_table(run_dir_a)
    summary_b = read_summary_table(run_dir_b)
    summary_json_a = read_summary_json(run_dir_a)
    summary_json_b = read_summary_json(run_dir_b)
    report_map_a = {}
    report_map_b = {}
    for strategy_name, strategy_id in STRATEGIES:
        report_map_a[strategy_name] = load_classification_report(
            summary_json_a, run_dir_a, strategy_name, strategy_id
        )
        report_map_b[strategy_name] = load_classification_report(
            summary_json_b, run_dir_b, strategy_name, strategy_id
        )

    best_a = select_best_strategy(summary_a)
    best_b = select_best_strategy(summary_b)

    comparison = build_comparison_table(summary_a, summary_b, report_map_a, report_map_b)
    loss_comparison = build_loss_comparison_table(summary_a, summary_b)
    best_only = build_best_only_table(summary_a, summary_b, best_a, best_b)

    out_pdf = os.path.join(outputs_dir, f"compare_{run_a}__{run_b}.pdf")
    with PdfPages(out_pdf) as pdf:
        title_fig = plt.figure(figsize=(11, 4))
        title_ax = title_fig.add_subplot(1, 1, 1)
        title_ax.axis("off")
        title_ax.text(0.5, 0.7, "Run Comparison", ha="center", fontsize=16)
        title_ax.text(0.5, 0.55, f"Run A: {run_a}", ha="center", fontsize=11)
        title_ax.text(0.5, 0.45, f"Run B: {run_b}", ha="center", fontsize=11)
        title_ax.text(0.5, 0.30, f"Best A: {best_a}", ha="center", fontsize=10)
        title_ax.text(0.5, 0.20, f"Best B: {best_b}", ha="center", fontsize=10)
        title_ax.text(
            0.5,
            0.08,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center",
            fontsize=9,
        )
        pdf.savefig(title_fig)
        plt.close(title_fig)

        best_table_fig = render_table_page(best_only, "Best Strategy Comparison")
        pdf.savefig(best_table_fig)
        plt.close(best_table_fig)

        table_fig = render_strategy_comparison_page(
            comparison,
            loss_comparison,
            "Strategy Comparison (Run A vs Run B)",
        )
        pdf.savefig(table_fig)
        plt.close(table_fig)

        # Best vs Best
        best_id_a = dict(STRATEGIES).get(best_a)
        best_id_b = dict(STRATEGIES).get(best_b)
        eval_a = load_eval(run_dir_a, best_id_a) if best_id_a else None
        eval_b = load_eval(run_dir_b, best_id_b) if best_id_b else None
        report_a = load_classification_report(summary_json_a, run_dir_a, best_a, best_id_a) if best_id_a else None
        report_b = load_classification_report(summary_json_b, run_dir_b, best_b, best_id_b) if best_id_b else None
        best_fig = render_cm_pair(eval_a, eval_b, f"Best vs Best ({best_a} vs {best_b})")
        pdf.savefig(best_fig)
        plt.close(best_fig)
        best_report_fig = render_report_comparison(
            report_a,
            report_b,
            f"Metrics - Best vs Best ({best_a} vs {best_b}) (test set)",
        )
        pdf.savefig(best_report_fig)
        plt.close(best_report_fig)

        # Per strategy
        for strategy_name, strategy_id in STRATEGIES:
            eval_a = load_eval(run_dir_a, strategy_id)
            eval_b = load_eval(run_dir_b, strategy_id)
            report_a = report_map_a.get(strategy_name)
            report_b = report_map_b.get(strategy_name)
            fig = render_cm_pair(eval_a, eval_b, strategy_name)
            pdf.savefig(fig)
            plt.close(fig)
            report_fig = render_report_comparison(
                report_a,
                report_b,
                f"Metrics - {strategy_name} (test set)",
            )
            pdf.savefig(report_fig)
            plt.close(report_fig)

    print(f"Wrote comparison PDF: {out_pdf}")


if __name__ == "__main__":
    main()
