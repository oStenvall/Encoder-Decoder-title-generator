import pdb

import torch

from models.EncoderDecoder import EncoderDecoder


class TitleGenerator(object):

    def __init__(self, src_vocab, tgt_vocab, attention,
                 hidden_dim, embedding_dim, bidirectional, device=torch.device('cpu')):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.i2w = {i: w for w, i in self.tgt_vocab.items()}
        self.device = device
        self.model = EncoderDecoder(len(src_vocab), len(tgt_vocab), attention
                                    , hidden_dim, embedding_dim, bidirectional).to(device)

    def generate_titles_with_attention(self, questions, max_len):

        # Run the decoder and convert the result into nested lists
        with torch.no_grad():
            decoded, alphas = tuple(d.cpu().numpy().tolist() for d in self.model.decode(questions, max_len))

        # Prune each decoded sentence after the first <eos>
        result = []
        dec_res = []
        dec_res_list = []
        for d, a in zip(decoded, alphas):
            d = [self.i2w[i] for i in d]
            try:
                eos_index = d.index('<eos>')
                del d[eos_index:]
                del a[eos_index:]
            except:
                pass
            dec_res.append([(' '.join(d))])
            dec_res_list.append(d)
            result.append((' '.join(d), a))
        return dec_res  # , result

    def generate_titles(self, questions, max_len):
        return self.generate_titles_with_attention(questions, max_len)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['enc_dec'])
        self.device = checkpoint['device']
        self.src_vocab = checkpoint['src_vocab']
        self.tgt_vocab = checkpoint['device']
        self.i2w = checkpoint['i2w']
