import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, vocab_size, attention, hidden_dim=128, embedding_dim=100, bidirectional_enc=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = attention
        if bidirectional_enc:
            gru_input_size = embedding_dim + 2 * hidden_dim
        else:
            gru_input_size = embedding_dim + hidden_dim
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(in_features=gru_input_size + hidden_dim, out_features=vocab_size)

    def forward(self, encoder_output, hidden, src_mask, tgt):
        batch_size, tgt_len = tgt.shape

        # Lookup the embeddings for the previous words
        embedded = self.embedding(tgt)

        # Initialise the list of outputs (in each sentence)
        outputs = []

        for i in range(tgt_len):
            # Get the embedding for the previous word (in each sentence)
            prev_embedded = embedded[:, i]


            # Take one step with the RNN
            output, hidden, alpha = self.step(encoder_output, hidden, src_mask, prev_embedded)

            # Update the list of outputs (in each sentence)
            outputs.append(output)

        return torch.cat(outputs, dim=1)

    def decode(self, encoder_output, hidden, src_mask, max_len):
        batch_size = encoder_output.size(0)

        # Initialise the list of generated words and attention weights (in each sentence)
        generated = [torch.ones(batch_size, dtype=torch.long, device=hidden.device)]
        alphas = []

        for i in range(max_len):
            # Get the embedding for the previous word (in each sentence)
            prev_embedded = self.embedding(generated[-1]).squeeze(dim=1)
            #print(prev_embedded.shape)

            # Take one step with the RNN
            output, hidden, alpha = self.step(encoder_output, hidden, src_mask, prev_embedded)

            # Update the list of generated words and attention weights (in each sentence)
            generated.append(output.argmax(-1))
            alphas.append(alpha)

        generated = [x.unsqueeze(1) for x in generated[1:]]
        alphas = [x.unsqueeze(1) for x in alphas]

        return torch.cat(generated, dim=1).squeeze(dim=2), torch.cat(alphas, dim=1)

    def step(self, encoder_output, hidden, src_mask, prev_embedded):

        context, alpha = self.attention.forward(decoder_hidden=hidden,
                                                encoder_output=encoder_output,
                                                src_mask=src_mask)
        embed_and_context = torch.cat((prev_embedded, context), dim=1).unsqueeze(1)
        gru_output, gru_hidden = self.gru(embed_and_context, hidden.unsqueeze(0))
        lin_input = torch.cat((gru_output, embed_and_context), dim=2)
        output = self.linear(lin_input)
        return output, gru_hidden.squeeze(0), alpha
