import torch
from torch import nn


class Method_RNN_Stage4(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, sequences):
        word_vectors = self.embedding(sequences)
        _, hidden = self.rnn(word_vectors)

        final_hidden = hidden[-1]
        return self.classifier(final_hidden)
