import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

DATA_SOURCE = "./eval_results/"
JSON_FILES = {
    "C1_clean": "eval_results_768_C1_clean_test.json",
    "C3_clean": "eval_results_768_C3_clean_test.json",
    "C3_noisy": "eval_results_768_C3_noisy_test.json",
    "N1_noisy": "eval_results_768_N1_noisy_test.json",
    "N3_noisy": "eval_results_768_N3_noisy_test.json",
    "N1_clean": "eval_results_768_N1_clean_test.json",
    "N3_clean": "eval_results_768_N3_clean_test.json",
}


def load_all_results(data_source, json_files):
    """Return a dict: {name: json_content}"""
    data = {}
    for name, filename in json_files.items():
        file_path = os.path.join(data_source, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data[name] = json.load(f)
    return data


def generate_confusion_matrix(json_obj, title, output_path=None):
    """Generate confusion matrix from a JSON object (not a file path!)"""
    details = json_obj["details"]

    y_true = [item["label"] for item in details]
    y_pred = [item["prediction"] for item in details]

    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)


    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=1,
        square=True
    )

    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.title(title + " (Counts)", fontsize=16)

    if output_path:
        plt.savefig(output_path.replace(".png", "_counts.png"),
                    dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path.replace('.png', '_counts.png')}")
    # plt.show()
    plt.close()


    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",          
        cmap="YlGnBu",
        linewidths=1,
        square=True,
        vmin=0.0,
        vmax=1.0
    )

    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.title(title + " (Row-normalized)", fontsize=16)

    if output_path:
        plt.savefig(output_path.replace(".png", "_normalized.png"),
                    dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path.replace('.png', '_normalized.png')}")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    output_dir = "./confusion_matrices"
    os.makedirs(output_dir, exist_ok=True)


    all_results = load_all_results(DATA_SOURCE, JSON_FILES)

    for name, content in all_results.items():
        print(f"Generating confusion matrix for {name}...")
        save_path = os.path.join(output_dir, f"{name}.png")
        title = f"Confusion Matrix for {name}"
        generate_confusion_matrix(content, title, save_path)
