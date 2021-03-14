import torch

from models.EncoderDecoder import EncoderDecoder


class QuestionAnswerer(object):

    def __init__(self, src_vocab, tgt_vocab, attention,
                 hidden_dim, embedding_dim, bidirectional, device=torch.device('cpu')):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.i2w = {i: w for w, i in self.tgt_vocab.items()}
        self.device = device
        self.model = EncoderDecoder(len(src_vocab), len(tgt_vocab), attention
                                    ,hidden_dim, embedding_dim, bidirectional).to(device)

    def generate_answers_with_attention(self, questions, max_len):
        # Encode each sentence
        #encoded = [[self.src_vocab.get(w, 3) for w in s.split()] for s in sentences]

        # Determine the maximal length of an encoded sentence
        #max_len = max(len(e) for e in encoded)

        # Build the input tensor, padding all sequences to the same length
        #src = torch.LongTensor([e + [0] * (max_len - len(e)) for e in encoded]).to(self.device)

        # Run the decoder and convert the result into nested lists
        with torch.no_grad():
            decoded, alphas = tuple(d.cpu().numpy().tolist() for d in self.model.decode(questions,  max_len))

        # Prune each decoded sentence after the first <eos>

        result = []
        for d, a in zip(decoded, alphas):
            d = [self.i2w[i] for i in d]
            try:
                eos_index = d.index('<eos>')
                del d[eos_index:]
                del a[eos_index:]
            except:
                pass
            result.append((' '.join(d), a))

        return result

    def generate_answers(self, questions, max_len):
        gen_answers, alphas = zip(*self.generate_answers_with_attention(questions, max_len))
        return gen_answers