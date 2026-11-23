import csv
import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class TextGenerationDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index], dtype=torch.long)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return sequence, target


def collate_batch(batch):
    sequences = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, targets


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def load_jokes(data_path=None, batch_size=32, max_vocab_size=10000):
    if data_path is None:
        data_path = Path(__file__).resolve().parents[2] / "data" / "stage_4_data" / "text_generation" / "data"

    jokes = []
    with Path(data_path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            words = tokenize(row["Joke"])
            if len(words) >= 2:
                jokes.append(words)

    indices = list(range(len(jokes)))
    random.Random(42).shuffle(indices)
    validation_count = max(1, int(len(indices) * 0.2))
    validation_indices = set(indices[:validation_count])
    training_jokes = [joke for i, joke in enumerate(jokes) if i not in validation_indices]
    validation_jokes = [joke for i, joke in enumerate(jokes) if i in validation_indices]

    counts = Counter(word for joke in training_jokes for word in joke)
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counts.most_common():
        if len(vocabulary) >= max_vocab_size:
            break
        vocabulary[word] = len(vocabulary)

    def make_examples(joke_list):
        inputs = []
        targets = []
        for joke in joke_list:
            token_ids = [vocabulary.get(word, vocabulary[UNK_TOKEN]) for word in joke]
            for position in range(1, len(token_ids)):
                inputs.append(token_ids[max(0, position - 30):position])
                targets.append(token_ids[position])
        return TextGenerationDataset(inputs, targets)

    training_data = make_examples(training_jokes)
    validation_data = make_examples(validation_jokes)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return training_loader, validation_loader, vocabulary, jokes
