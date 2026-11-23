import re
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def read_reviews(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "stage_4_data" / "text_classification"
    data_dir = Path(data_dir)

    reviews = {}
    for split in ("train", "test"):
        split_reviews = []
        split_labels = []
        for folder, label in (("pos", 1), ("neg", 0)):
            for file_path in sorted((data_dir / split / folder).glob("*.txt")):
                text = file_path.read_text(encoding="utf-8", errors="replace")
                split_reviews.append(tokenize(text))
                split_labels.append(label)
        reviews[split] = (split_reviews, split_labels)
    return reviews


def build_vocabulary(training_reviews, max_vocab_size=None):
    word_counts = Counter(word for review in training_reviews for word in review)
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in word_counts.most_common():
        if max_vocab_size is not None and len(vocabulary) >= max_vocab_size:
            break
        vocabulary[word] = len(vocabulary)
    return vocabulary


def encode_reviews(reviews, vocabulary, max_length=500):
    sequences = []
    for review in reviews:
        token_ids = [vocabulary.get(word, vocabulary[UNK_TOKEN]) for word in review[:max_length]]
        token_ids += [vocabulary[PAD_TOKEN]] * (max_length - len(token_ids))
        sequences.append(token_ids)
    return torch.tensor(sequences, dtype=torch.long)


def load_data(data_dir=None, max_vocab_size=None, max_length=500, batch_size=32,
              use_validation=False):
    reviews = read_reviews(data_dir)
    train_reviews, train_labels = reviews["train"]
    test_reviews, test_labels = reviews["test"]

    if use_validation:
        random_generator = random.Random(42)
        train_indices = []
        validation_indices = []
        labels_by_value = {}

        for index, label in enumerate(train_labels):
            if label not in labels_by_value:
                labels_by_value[label] = []
            labels_by_value[label].append(index)

        for label_indices in labels_by_value.values():
            random_generator.shuffle(label_indices)
            validation_count = int(len(label_indices) * 0.2)
            validation_indices.extend(label_indices[:validation_count])
            train_indices.extend(label_indices[validation_count:])

        training_reviews = [train_reviews[index] for index in train_indices]
        training_labels = [train_labels[index] for index in train_indices]
        validation_reviews = [train_reviews[index] for index in validation_indices]
        validation_labels = [train_labels[index] for index in validation_indices]
    else:
        training_reviews = train_reviews
        training_labels = train_labels

    vocabulary = build_vocabulary(training_reviews, max_vocab_size)
    train_sequences = encode_reviews(training_reviews, vocabulary, max_length)
    test_sequences = encode_reviews(test_reviews, vocabulary, max_length)

    train_data = TensorDataset(train_sequences, torch.tensor(training_labels, dtype=torch.long))
    test_data = TensorDataset(test_sequences, torch.tensor(test_labels, dtype=torch.long))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if use_validation:
        validation_sequences = encode_reviews(validation_reviews, vocabulary, max_length)
        validation_data = TensorDataset(
            validation_sequences, torch.tensor(validation_labels, dtype=torch.long)
        )
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader, test_loader, vocabulary

    return train_loader, test_loader, vocabulary
