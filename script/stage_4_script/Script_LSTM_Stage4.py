import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


RANDOM_SEED = 42
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_RNN_LAYERS = 1


def main():
    project_dir = Path(__file__).resolve().parents[2]
    stage_code_dir = project_dir / "local_code" / "stage_4_code"
    sys.path.insert(0, str(stage_code_dir))
    from Dataset_Loader_Stage4 import load_data
    from Method_LSTM_Stage4 import Method_LSTM_Stage4

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader, _, vocabulary = load_data(
        max_vocab_size=MAX_VOCAB_SIZE,
        max_length=MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        use_validation=True,
    )
    model = Method_LSTM_Stage4(
        len(vocabulary), EMBEDDING_DIM, HIDDEN_DIM, NUM_RNN_LAYERS
    ).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    result_dir = project_dir / "result" / "stage_4_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for reviews, labels in train_loader:
            reviews = reviews.to(device)
            labels = labels.to(device)

            lengths = (reviews != 0).sum(dim=1).clamp(min=1)
            batch_length = int(lengths.max().item())
            reviews = reviews[:, :batch_length]
            word_vectors = model.embedding(reviews)
            lstm_outputs, _ = model.lstm(word_vectors)
            row_indices = torch.arange(reviews.size(0), device=device)
            last_outputs = lstm_outputs[row_indices, lengths - 1]
            scores = model.classifier(last_outputs)

            loss = loss_function(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total
        training_losses.append(epoch_loss)
        model.eval()
        validation_total_loss = 0.0
        validation_correct = 0
        validation_total = 0
        with torch.no_grad():
            for reviews, labels in validation_loader:
                reviews = reviews.to(device)
                labels = labels.to(device)

                lengths = (reviews != 0).sum(dim=1).clamp(min=1)
                batch_length = int(lengths.max().item())
                reviews = reviews[:, :batch_length]
                word_vectors = model.embedding(reviews)
                lstm_outputs, _ = model.lstm(word_vectors)
                row_indices = torch.arange(reviews.size(0), device=device)
                last_outputs = lstm_outputs[row_indices, lengths - 1]
                scores = model.classifier(last_outputs)
                loss = loss_function(scores, labels)

                validation_total_loss += loss.item() * labels.size(0)
                predictions = scores.argmax(dim=1)
                validation_correct += (predictions == labels).sum().item()
                validation_total += labels.size(0)

        validation_loss = validation_total_loss / validation_total
        validation_accuracy = validation_correct / validation_total
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"training loss: {epoch_loss:.4f}, "
            f"training accuracy: {epoch_accuracy:.4f}, "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), result_dir / "lstm_model_weights.pth")

    plt.plot(range(1, EPOCHS + 1), training_losses, marker="o", label="Training loss")
    plt.plot(range(1, EPOCHS + 1), validation_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "lstm_training_loss.png")
    plt.close()

    config = {
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_rnn_layers": NUM_RNN_LAYERS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
    }
    with (result_dir / "lstm_training_config_and_metrics.json").open("w") as file:
        json.dump(config, file, indent=2)


if __name__ == "__main__":
    main()
