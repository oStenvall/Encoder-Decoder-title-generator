import torch
from torch.utils.data import Dataset


class TitleQuestionTestDataset(Dataset):
    def __init__(self, src_vocab, src_sequences, tgt_sequences):
        self.src_vocab = src_vocab

        # <pad> (0) <bos> (1), <eos> (2), and <unk> (3).
        self.src_sequences = src_sequences
        self.encoder_input = [[self.src_vocab.get(w, 3) for w in s] for s in src_sequences]
        self.ref_sentence = [" ".join(s) for s in tgt_sequences]

    def __getitem__(self, idx):
        return torch.LongTensor(self.encoder_input[idx]).unsqueeze(0), self.ref_sentence[idx]

    def __len__(self):
        assert len(self.encoder_input) == len(self.ref_sentence)
        return len(self.encoder_input)

    def get_src_sequence_by_idx(self, idx):
        return self.src_sequences[idx]

    def create_sample_tensor(self, question, size):
        assert len(question) <= 10
        if len(question) < 10:
            src = [self.src_vocab.get(w, 3) for w in question]
            padding = (size - len(question)) * [self.src_vocab["<pad>"]]
            src = src + padding
        else:
            src = [self.src_vocab.get(w, 3) for w in question]

        return torch.LongTensor(src).unsqueeze(0)

    def example(self, i):
        src, tgt_sentence = self[i]
        return torch.LongTensor(src).unsqueeze(0), tgt_sentence

    def get_all_examples(self):
        return torch.LongTensor(self.encoder_input), self.ref_sentence
