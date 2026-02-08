import json
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
from ChestXrayDataset import VALID_IMAGE_EXTS

def count_images(data_dir, class_names):
    counts = {}
    total = 0
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        n = 0
        for name in os.listdir(cls_path):
            path = os.path.join(cls_path, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(name)
            if ext.lower() not in VALID_IMAGE_EXTS:
                continue
            n += 1
        counts[cls] = n
        total += n
    return counts, total


def display_images_from_dataset(dataset, n=10):
    index_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    for _ in range(n):
        idx = random.randint(0, len(dataset)-1)
        image, label = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = image.clip(0, 1)
        plt.imshow(image)
        plt.title(f"Label: {index_to_class[label]}")
        plt.axis('off')
        plt.show()


def plot_metrics(train_losses, val_losses, train_accs, val_accs, title_suffix="", show=False):
    fig_loss = plt.figure(figsize=(10,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss {title_suffix}")
    plt.legend()
    if show:
        plt.show()

    fig_acc = plt.figure(figsize=(10,4))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training vs Validation Accuracy {title_suffix}")
    plt.legend()
    if show:
        plt.show()
    return fig_loss, fig_acc

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", show=False):
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    if show:
        plt.show()
    return fig

def report(
    experiment_name,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    eval_labels,
    eval_preds,
    class_names,
    persist=True,
    output_dir="outputs",
    classification_report_text=None,
):
    if not persist:
        return

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{experiment_name}-reports.pdf")
    with PdfPages(report_path) as pdf:
        fig_loss, fig_acc = plot_metrics(train_losses, val_losses, train_accs, val_accs, experiment_name, show=False)
        pdf.savefig(fig_loss)
        pdf.savefig(fig_acc)
        plt.close(fig_loss)
        plt.close(fig_acc)

        cm = confusion_matrix(eval_labels, eval_preds)
        fig_cm = plot_confusion_matrix(cm, class_names, title=f"Confusion Matrix - {experiment_name}", show=False)
        pdf.savefig(fig_cm)
        plt.close(fig_cm)

        report_text = classification_report(eval_labels, eval_preds, target_names=class_names)
        print(report_text)
        if classification_report_text:
            fig_text = plt.figure(figsize=(8.5, 6))
            plt.axis("off")
            plt.text(
                0.01,
                0.99,
                classification_report_text,
                fontfamily="monospace",
                fontsize=9,
                va="top",
            )
            pdf.savefig(fig_text)
            plt.close(fig_text)


def write_classification_report(output_dir, strategy_id, labels, preds, class_names):
    report_dict = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=False,
        zero_division=0,
    )
    report_path = os.path.join(output_dir, f"classification_report_{strategy_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    return report_dict, report_text


def compare_training_strategies(
    cfg_name,
    cfg,
    optimizer_name,
    weight_decay_backbone,
    weight_decay_head,
    timestamp,
    run_tag,
    run_name,
    output_dir,
    batch_size,
    num_workers,
    report_dict_frozen,
    report_dict_full,
    report_dict_partial,
    strategies,
    test_accuracies,
    test_losses,
    final_train_loss,
    final_val_loss,
    final_train_acc,
    final_val_acc,
    overfit_gap,
):
    summary_df = pd.DataFrame({
        'Strategy': strategies,
        'Train Accuracy': final_train_acc,
        'Train Loss': final_train_loss,
        'Test Accuracy': test_accuracies,
        'Test Loss': test_losses,
        'Val Accuracy': final_val_acc,
        'Val Loss': final_val_loss,
    })

    print("=== Comparative Analysis Summary ===")
    print(summary_df)

    summary_path = os.path.join(output_dir, "summary.txt")
    summary_json_path = os.path.join(output_dir, "summary.json")
    run_params = {
        "cfg_name": cfg_name,
        "run_tag": run_tag or None,
        "timestamp": timestamp,
        "optimizer": optimizer_name,
        "weight_decay_backbone": weight_decay_backbone,
        "weight_decay_head": weight_decay_head,
        "lr_frozen": cfg["lr_frozen"],
        "lr_full": cfg["lr_full"],
        "lr_partial": cfg["lr_partial"],
        "epochs_frozen": cfg["epochs_frozen"],
        "epochs_full": cfg["epochs_full"],
        "epochs_partial": cfg["epochs_partial"],
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_df.to_string(index=True))
        f.write("\n\n")
        f.write(f"Run: {run_name}\n")
        f.write("Params:\n")
        for key, value in run_params.items():
            f.write(f"- {key}: {value}\n")

    def json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    metrics = {}
    for idx, strategy in enumerate(strategies):
        row = summary_df.iloc[idx]
        metrics[strategy] = {
            col: json_safe(row[col]) for col in summary_df.columns if col != "Strategy"
        }

    summary_json = {
        "run": {
            "name": run_name,
            "timestamp": timestamp,
            "cfg_name": cfg_name,
        },
        "params": {k: json_safe(v) for k, v in run_params.items()},
        "metrics": metrics,
        "classification_reports": {
            "FC Only": report_dict_frozen,
            "Full Fine-Tuning": report_dict_full,
            "Partial Fine-Tuning": report_dict_partial,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    for i, acc in enumerate(test_accuracies):
        if i == 0:
            note = "Fastest training, lowest resource usage, may underfit new data."
        elif i == 1:
            note = "Highest accuracy, full model adaptation, but slowest and most GPU intensive."
        else:
            note = "Balanced approach, moderate accuracy and resource usage."

        print(f"{strategies[i]}:\n"
                f"  Test Accuracy = {acc:.4f}\n"
                f"  Test Loss = {test_losses[i]:.4f}\n"
                f"  Final Train Loss = {final_train_loss[i]:.4f}\n"
                f"  Final Val Loss = {final_val_loss[i]:.4f}\n"
                f"  Overfit Gap = {overfit_gap[i]:.4f}\n"
                f"  Note: {note}\n")
