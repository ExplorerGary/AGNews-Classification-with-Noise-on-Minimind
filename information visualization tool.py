# information visualization tool
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
COLOR_MAP = {
    "C1": "#1f77b4",  
    "C3": "#ff7f0e",  
    "N1": "#2ca02c",  
    "N3": "#d62728",  
}

"""
EXAMPLE JSON FILE CONTENT:
{
  "accuracy": 0.5269736842105263,
  "precision": 0.5253670698824053,
  "recall": 0.5269736842105264,
  "f1": 0.5260162769450847,
  "details": [
    {
      "input": "Classify the following news into one of the categories (0=World, 1=Sports, 2=Business, 3=Sci/Tech):\nfears for t n pension after talks unions representing workers at turner newall say they are disappointed after talks with stricken parent firm federal mogul",
      "label": 2,
      "prediction": 1,
      "raw_output": "1"
    },

"""


# this tool is used to generate visual representations of data, which is used in the presentation



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
print("Information Visualization Tool Loaded")




def load_all_results(data_source, json_files):
    data = {}
    for k, v in json_files.items():
        with open(data_source + v, "r", encoding="utf-8") as f:
            data[k] = json.load(f)
    return data


# 1. the difference between C1 and C3 on clean, we will use it show the potential of GPT in classification.
def plot_c1_c3_difference(data):
    labels = ['C1 Clean', 'C3 Clean']
    accuracies = [
        data['C1_clean']['accuracy'],
        data['C3_clean']['accuracy']
    ]
    colors = [COLOR_MAP["C1"], COLOR_MAP["C3"]]

    plt.figure(figsize=(8,9))  
    plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 1)
    plt.title('Impact of Fine-Tuning on Clean Data')
    plt.ylabel('Accuracy')

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', fontsize = 64)

    plt.tight_layout()
    plt.savefig("fig_c1_vs_c3_clean.png", dpi=200)
    plt.close()



# 2. the impact of noise on C3, we will use it to show the robustness of GPT in classification.
def plot_c3_noise_impact(data):
    labels = ['C3 Clean', 'C3 Noisy']
    accuracies = [
        data['C3_clean']['accuracy'],
        data['C3_noisy']['accuracy']
    ]
    colors = [COLOR_MAP["C3"], COLOR_MAP["C3"]]

    plt.figure(figsize=(8,9)) 
    plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 1)
    plt.title('Impact of Noise on C3 Accuracy')
    plt.ylabel('Accuracy')

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', fontsize = 64)

    plt.tight_layout()
    plt.savefig("fig_c3_clean_vs_noisy.png", dpi=200)
    plt.close()


# 3. the impact of noise on N1 and N3, we will use it to show the impact of noise on both noisy and clean inputs.
def plot_c3_n1_n3_comparison(data, mode="clean"):
    if mode == "clean":
        labels = ['C3 Clean', 'N1 Clean', 'N3 Clean']
        accuracies = [
            data['C3_clean']['accuracy'],
            data['N1_clean']['accuracy'],
            data['N3_clean']['accuracy']
        ]
        colors = [COLOR_MAP["C3"], COLOR_MAP["N1"], COLOR_MAP["N3"]]
        filename = "fig_c3_n1_n3_clean.png"
        title = "Performance on Clean Data after Noisy Fine-Tuning"
    else:
        labels = ['C3 Noisy', 'N1 Noisy', 'N3 Noisy']
        accuracies = [
            data['C3_noisy']['accuracy'],
            data['N1_noisy']['accuracy'],
            data['N3_noisy']['accuracy']
        ]
        colors = [COLOR_MAP["C3"], COLOR_MAP["N1"], COLOR_MAP["N3"]]
        filename = "fig_c3_n1_n3_noisy.png"
        title = "Performance on Noisy Data after Noisy Fine-Tuning"

    plt.figure(figsize=(8, 9))  
    plt.bar(labels, accuracies, color=colors)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Accuracy')

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', fontsize = 64)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



if __name__ == "__main__":
    data = load_all_results(DATA_SOURCE, JSON_FILES)

    plot_c1_c3_difference(data)
    plot_c3_noise_impact(data)

    plot_c3_n1_n3_comparison(data, mode="clean")
    plot_c3_n1_n3_comparison(data, mode="noisy")

    print("All figures generated successfully.")
