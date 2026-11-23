import csv
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def main():
    project_dir = Path(__file__).resolve().parents[2]
    dataset_path = project_dir / "data" / "stage_4_data" / "text_generation" / "data"
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        jokes = [row["Joke"] for row in csv.DictReader(file) if row.get("Joke")]

    training_jokes, _ = train_test_split(
        jokes, test_size=0.2, random_state=RANDOM_SEED
    )
    selected_jokes = random.Random(RANDOM_SEED).sample(training_jokes, 5)

    print("Five original jokes from the training portion:")
    for number, joke in enumerate(selected_jokes, start=1):
        print(f"{number}. {joke}")

    result_dir = project_dir / "result" / "stage_4_result" / "text_generation"
    model_files = [
        ("RNN", result_dir / "generated_jokes.txt"),
        ("LSTM", result_dir / "lstm" / "generated_jokes.txt"),
        ("GRU", result_dir / "gru" / "generated_jokes.txt"),
    ]
    generated_examples = [
        (name, path.read_text(encoding="utf-8").strip())
        for name, path in model_files
    ]

    comparison = ["Training jokes vs. generated examples", ""]
    comparison.append("Five original training jokes:")
    comparison.extend(
        f"{number}. {joke}" for number, joke in enumerate(selected_jokes, start=1)
    )
    for name, examples in generated_examples:
        comparison.extend(["", f"{name} generated examples:", examples])

    output_path = result_dir / "training_vs_generated_comparison.txt"
    output_path.write_text("\n".join(comparison) + "\n", encoding="utf-8")
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
