import torch
from torch.utils.data import Dataset


class QuestionAnswerTestDataset(Dataset):
    def __init__(self, src_vocab, src_sequences, tgt_sequences):
        self.src_vocab = src_vocab

        # <pad> (0) <bos> (1), <eos> (2), and <unk> (3).
        self.encoder_input = [[self.src_vocab.get(w, 3) for w in s] for s in src_sequences]
        self.target_sentence = [" ".join(s) for s in tgt_sequences]

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.target_sentence[idx]

    def __len__(self):
        assert len(self.encoder_input) == len(self.target_sentence)
        return len(self.encoder_input)

    def create_sample_tensor(self, question, size):
        assert len(question) <= 10
        if len(question) < 10:
            src = [self.src_vocab.get(w, 3) for w in question]
            padding = (size - question) * [self.src_vocab["<pad>"]]
            src = [src + padding]
        else:
            src = [[self.src_vocab.get(w, 3) for w in question]]

        return torch.LongTensor(src).unsqueeze(0)

    def example(self, i):
        src, tgt_sentence = self[i]
        return torch.LongTensor(src).unsqueeze(0), tgt_sentence

    def get_all_examples(self):
        return torch.LongTensor(self.encoder_input), self.target_sentence



