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
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
MAX_GENERATED_WORDS = 25
PROMPTS = ["what did the", "don't you hate", "two artists had"]


def generate_text(model, prompt, vocabulary, device, generator):
    words = prompt.lower().split()
    word_ids = [vocabulary.get(word, vocabulary["<UNK>"]) for word in words]
    model.eval()
    for _ in range(MAX_GENERATED_WORDS):
        sequence = torch.tensor([word_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model(sequence).squeeze(0).cpu()
        probabilities = torch.softmax(scores / 0.8, dim=0)
        probabilities[vocabulary["<PAD>"]] = 0
        probabilities[vocabulary["<UNK>"]] = 0
        if probabilities.sum().item() == 0:
            break
        next_id = torch.multinomial(probabilities, 1, generator=generator).item()
        word_ids.append(next_id)
    reverse_vocabulary = {index: word for word, index in vocabulary.items()}
    return " ".join(reverse_vocabulary.get(index, "<UNK>") for index in word_ids)


def train_model(model_name, model_class, training_loader, validation_loader, vocabulary, jokes, project_dir, device):
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    model = model_class(len(vocabulary), EMBEDDING_DIM, HIDDEN_DIM).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    result_dir = project_dir / "result" / "stage_4_result" / "text_generation" / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = result_dir / "model_weights.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for sequences, targets in training_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            loss = loss_function(model(sequences), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_examples += targets.size(0)
        training_loss = total_loss / total_examples
        training_losses.append(training_loss)

        model.eval()
        validation_total_loss = 0.0
        validation_examples = 0
        with torch.no_grad():
            for sequences, targets in validation_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                loss = loss_function(model(sequences), targets)
                validation_total_loss += loss.item() * targets.size(0)
                validation_examples += targets.size(0)
        validation_loss = validation_total_loss / validation_examples
        validation_losses.append(validation_loss)
        print(f"{model_name.upper()} epoch {epoch + 1}/{EPOCHS} - training loss: {training_loss:.4f}, validation loss: {validation_loss:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

    plt.plot(range(1, EPOCHS + 1), training_losses, marker="o", label="Training loss")
    plt.plot(range(1, EPOCHS + 1), validation_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Text Generation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "training_loss.png")
    plt.close()

    metrics = {
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "training_examples": len(training_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
    }
    with (result_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    generated_examples = [
        f"Prompt: {prompt}\nGenerated: {generate_text(model, prompt, vocabulary, device, generator)}"
        for prompt in PROMPTS
    ]
    (result_dir / "generated_jokes.txt").write_text(
        "\n\n".join(generated_examples) + "\n", encoding="utf-8"
    )
    return best_validation_loss


def main():
    project_dir = Path(__file__).resolve().parents[2]
    stage_code_dir = project_dir / "local_code" / "stage_4_code"
    sys.path.insert(0, str(stage_code_dir))
    from Dataset_Loader_TextGeneration_Stage4 import load_jokes
    from Method_GRU_TextGeneration_Stage4 import Method_GRU_TextGeneration_Stage4
    from Method_LSTM_TextGeneration_Stage4 import Method_LSTM_TextGeneration_Stage4

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader, validation_loader, vocabulary, jokes = load_jokes(
        batch_size=BATCH_SIZE, max_vocab_size=MAX_VOCAB_SIZE
    )
    lstm_loss = train_model(
        "lstm", Method_LSTM_TextGeneration_Stage4, training_loader,
        validation_loader, vocabulary, jokes, project_dir, device
    )
    gru_loss = train_model(
        "gru", Method_GRU_TextGeneration_Stage4, training_loader,
        validation_loader, vocabulary, jokes, project_dir, device
    )
    print(f"Best validation loss comparison - LSTM: {lstm_loss:.4f}, GRU: {gru_loss:.4f}")


if __name__ == "__main__":
    main()
