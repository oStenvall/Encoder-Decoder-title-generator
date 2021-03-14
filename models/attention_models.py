import torch.nn.functional as F
import torch
import torch.nn as nn


class UniformAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_output, src_mask):
        batch_size, src_len, _ = encoder_output.shape

        # Set all attention scores to the same constant value (0). After
        # the softmax, we will have uniform weights.
        scores = torch.zeros(batch_size, src_len, device=encoder_output.device)

        # Mask out the attention scores for the padding tokens. We set
        # them to -inf. After the softmax, we will have 0.
        scores.data.masked_fill_(~src_mask, -float('inf'))

        # Convert scores into weights
        alpha = F.softmax(scores, dim=1)

        # The context is the alpha-weighted sum of the encoder outputs.
        context = torch.bmm(alpha.unsqueeze(1), encoder_output).squeeze(1)

        return context, alpha


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_dim=128, bidirectional_enc=True):
        super().__init__()
        self.linear_W = nn.Linear(in_features=hidden_dim,out_features= hidden_dim)
        self.bidirectional_enc = bidirectional_enc
        if bidirectional_enc:
            self.linear_U = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim)
        else:
            self.linear_U = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.tanh = nn.Tanh()

    def forward(self, decoder_hidden, encoder_output, src_mask):

        u = self.linear_U(encoder_output)
        w = self.linear_W(decoder_hidden).unsqueeze(1)

        tan_w_u = self.tanh(w + u)
        scores = tan_w_u @ self.v

        # The rest of the code is as in UniformAttention
        # Mask out the attention scores for the padding tokens. We set
        # them to -inf. After the softmax, we will have 0.
        scores.data.masked_fill_(~src_mask, -float('inf'))

        # Convert scores into weights
        alpha = F.softmax(scores, dim=1)
        # The context vector is the alpha-weighted sum of the encoder outputs.
        context = torch.bmm(alpha.unsqueeze(1), encoder_output).squeeze(1)

        return context, alpha
