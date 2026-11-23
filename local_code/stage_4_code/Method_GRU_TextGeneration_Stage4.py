import torch
from torch import nn


class Method_GRU_TextGeneration_Stage4(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequences):
        word_vectors = self.embedding(sequences)
        outputs, _ = self.gru(word_vectors)
        lengths = (sequences != 0).sum(dim=1).clamp(min=1)
        rows = torch.arange(sequences.size(0), device=sequences.device)
        final_outputs = outputs[rows, lengths - 1]
        return self.output(final_outputs)
