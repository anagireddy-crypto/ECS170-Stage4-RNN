import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    project_dir = Path(__file__).resolve().parents[2]
    stage_code_dir = project_dir / "local_code" / "stage_4_code"
    sys.path.insert(0, str(stage_code_dir))
    from Dataset_Loader_Stage4 import load_data
    from Method_GRU_Stage4 import Method_GRU_Stage4
    from Method_LSTM_Stage4 import Method_LSTM_Stage4
    from Method_RNN_Stage4 import Method_RNN_Stage4

    _, _, test_loader, vocabulary = load_data(
        max_vocab_size=10000,
        max_length=200,
        batch_size=32,
        use_validation=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_settings = [
        (
            "RNN",
            Method_RNN_Stage4,
            "rnn_validation_baseline/rnn_model_weights.pth",
            "rnn",
        ),
        (
            "LSTM",
            Method_LSTM_Stage4,
            "lstm_validation_baseline/lstm_model_weights.pth",
            "lstm",
        ),
        (
            "GRU",
            Method_GRU_Stage4,
            "gru_validation_baseline/gru_model_weights.pth",
            "gru",
        ),
    ]
    results = {}

    for name, model_class, weights_path, recurrent_layer in model_settings:
        model = model_class(
            len(vocabulary), embedding_dim=100, hidden_dim=128, num_layers=1
        ).to(device)
        model.load_state_dict(
            torch.load(
                project_dir / "result" / "stage_4_result" / weights_path,
                map_location=device,
            )
        )
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for reviews, labels in test_loader:
                reviews = reviews.to(device)
                lengths = (reviews != 0).sum(dim=1).clamp(min=1)
                batch_length = int(lengths.max().item())
                reviews = reviews[:, :batch_length]
                word_vectors = model.embedding(reviews)
                outputs, _ = getattr(model, recurrent_layer)(word_vectors)
                row_indices = torch.arange(reviews.size(0), device=device)
                last_outputs = outputs[row_indices, lengths - 1]
                scores = model.classifier(last_outputs)
                predictions = scores.argmax(dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.cpu().tolist())

        results[name] = {
            "accuracy": accuracy_score(all_labels, all_predictions),
            "weighted_precision": precision_score(
                all_labels, all_predictions, average="weighted", zero_division=0
            ),
            "weighted_recall": recall_score(
                all_labels, all_predictions, average="weighted", zero_division=0
            ),
            "weighted_f1": f1_score(
                all_labels, all_predictions, average="weighted", zero_division=0
            ),
        }

    print("Test set classification results")
    print(f"{'Model':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name, metrics in results.items():
        print(
            f"{name:<8} {metrics['accuracy']:>10.4f} "
            f"{metrics['weighted_precision']:>10.4f} "
            f"{metrics['weighted_recall']:>10.4f} "
            f"{metrics['weighted_f1']:>10.4f}"
        )

    output_path = project_dir / "result" / "stage_4_result" / "final_classification_metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
