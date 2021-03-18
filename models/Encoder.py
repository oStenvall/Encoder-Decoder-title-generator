import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, GRU


class Encoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim=128, embedding_dim=100, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        else:
            self.linear = Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x):
        embed = self.embedding(x)
        if self.bidirectional:
            output, hidden = self.gru(embed)
            hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
            hidden = self.linear(hidden)
            return output, hidden
        else:
            output, hidden = self.gru(embed)
            hidden = self.linear(hidden).squeeze(0)
            return output, hidden
