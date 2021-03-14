import torch
from torch.utils.data import Dataset


class QuestionAnswerTrainDataset(Dataset):
    def __init__(self, src_vocab, tgt_vocab, src_sequences, tgt_input_sequences, tgt_output_sequences):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # <pad> (0) <bos> (1), <eos> (2), and <unk> (3).
        self.encoder_input = [[self.src_vocab.get(w, 3) for w in s] for s in src_sequences]
        self.decoder_input = [[self.tgt_vocab.get(w, 3) for w in s] for s in tgt_input_sequences]
        self.decoder_target = [[self.tgt_vocab.get(w, 3) for w in s] for s in tgt_output_sequences]

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_target[idx]

    def __len__(self):
        assert len(self.encoder_input) == len(self.decoder_input) == len(self.decoder_target)
        return len(self.encoder_input)



def example(dataset, i):
    src, tgt, _ = dataset[i]
    return torch.LongTensor(src).unsqueeze(0), torch.LongTensor(tgt).unsqueeze(0)