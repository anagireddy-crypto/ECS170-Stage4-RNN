import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "result" / "stage_4_result"
FIGURE_DIR = RESULT_DIR / "report_figures"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_text_generation_loss():
    model_files = {
        "RNN": RESULT_DIR / "text_generation" / "text_generation_metrics.json",
        "LSTM": RESULT_DIR / "text_generation" / "lstm" / "training_metrics.json",
        "GRU": RESULT_DIR / "text_generation" / "gru" / "training_metrics.json",
    }
    plt.figure(figsize=(8, 5))
    for name, path in model_files.items():
        metrics = load_json(path)
        losses = metrics["validation_losses"]
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, marker="o", label=name)
    plt.title("Text Generation Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.xticks(range(1, 6))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "text_generation_loss.png", dpi=160)
    plt.close()


def plot_ablation_comparison():
    experiments = {
        "128": RESULT_DIR / "rnn_validation_baseline" / "training_config_and_metrics.json",
        "256": RESULT_DIR / "rnn_hidden_256" / "training_config_and_metrics.json",
    }
    hidden_sizes = []
    losses = []
    accuracies = []
    for size, path in experiments.items():
        metrics = load_json(path)
        hidden_sizes.append(size)
        losses.append(metrics["best_validation_loss"])
        accuracies.append(metrics["best_validation_accuracy"])

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(hidden_sizes, losses, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Best Validation Loss")
    axes[0].set_xlabel("RNN hidden size")
    axes[0].set_ylabel("Loss")
    axes[1].bar(hidden_sizes, accuracies, color=["#4C78A8", "#F58518"])
    axes[1].set_title("Accuracy at Best-Loss Epoch")
    axes[1].set_xlabel("RNN hidden size")
    axes[1].set_ylabel("Validation accuracy")
    figure.suptitle("RNN Hidden-Size Ablation")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "ablation_comparison.png", dpi=160)
    plt.close(figure)


def make_architecture_diagram():
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")

    labels = ["Input text", "Tokenization", "Embedding", "RNN / LSTM / GRU"]
    x_positions = [0.5, 2.7, 4.9, 7.1]
    for label, x in zip(labels, x_positions):
        axis.text(x + 0.8, 3.7, label, ha="center", va="center", fontsize=11,
                  bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E8EEF7", "edgecolor": "#4C78A8"})
    for x in [2.35, 4.55, 6.75]:
        axis.annotate("", xy=(x + 0.25, 3.7), xytext=(x, 3.7),
                      arrowprops={"arrowstyle": "->", "lw": 1.5})

    output_boxes = [
        ("Classification output layer", "2 sentiment classes", 4.5, "#E6F4EA", "#54A24B"),
        ("Generation output layer", "Next-word prediction", 1.5, "#FCE8E6", "#E45756"),
    ]
    for title, subtitle, y, fill, edge in output_boxes:
        axis.annotate("", xy=(9.3, y), xytext=(9.3, 3.55),
                      arrowprops={"arrowstyle": "->", "lw": 1.5, "color": edge})
        axis.text(10.5, y, f"{title}\n{subtitle}", ha="center", va="center", fontsize=10,
                  bbox={"boxstyle": "round,pad=0.55", "facecolor": fill, "edgecolor": edge})
    axis.set_title("Stage 4 Sequence Model Architecture", fontsize=15, pad=15)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "model_architecture.png", dpi=160)
    plt.close(figure)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_text_generation_loss()
    plot_ablation_comparison()
    make_architecture_diagram()
    print(f"Saved supported report figures to {FIGURE_DIR}")
    print("Classification epoch curves were not created: per-epoch values are absent from saved metrics.")


if __name__ == "__main__":
    main()
