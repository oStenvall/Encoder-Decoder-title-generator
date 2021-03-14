from torch import nn
from models.Decoder import Decoder
from models.Encoder import Encoder


class EncoderDecoder(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, attention, hidden_dim, embedding_dim, bidirectional=True):
        super().__init__()
        self.encoder = Encoder(vocab_size=src_vocab_size, hidden_dim=hidden_dim,
                               embedding_dim=embedding_dim, bidirectional=bidirectional)
        self.decoder = Decoder(vocab_size=tgt_vocab_size, attention=attention,
                               hidden_dim=hidden_dim, embedding_dim=embedding_dim, bidirectional_enc=bidirectional)

    def forward(self, src, tgt):
        encoder_output, hidden = self.encoder(src)
        return self.decoder.forward(encoder_output, hidden, src != 0, tgt)

    def decode(self, src, max_len):
        encoder_output, hidden = self.encoder(src)
        print("Encoder output: " + str(encoder_output.shape))
        print("Encoder hidden: " + str(hidden.shape))
        print("Max len: " + str(max_len))
        return self.decoder.decode(encoder_output, hidden, src != 0, max_len)