import pickle

import torch

from dataset.QuestionAnswerTestDataset import QuestionAnswerTestDataset
from dataset.QuestionAnswerTrainDataset import QuestionAnswerTrainDataset
from models.attention_models import UniformAttention
from test_functions import print_encoder_decoder_shape, BahdanauAttention, test_encoder_decoder_model, test_QnA_bot
from models.EncoderDecoder import EncoderDecoder
from models.QuestionAnswerer import QuestionAnswerer
from train import train


def main():
    file = open('data/data.p', "rb")
    data_and_vocab = pickle.load(file)
    src_vocab = data_and_vocab["src_vocab"]
    tgt_vocab = data_and_vocab["tgt_vocab"]
    assert len(src_vocab) == len(tgt_vocab)
    questions = data_and_vocab["questions"]
    answer_inputs = data_and_vocab["answer_inputs"]
    answer_targets = data_and_vocab["answer_targets"]
    ref_a = data_and_vocab["ref_a"]
    q_train = questions[:25000]
    a_train_input = answer_inputs[:25000]

    q_val_input = questions[25000:]
    #a_val_tgt = answer_targets[25000:]
    ref_a_val = ref_a[25000:]

    val_dataset = QuestionAnswerTestDataset(src_vocab, q_val_input, ref_a_val)
    train_dataset = QuestionAnswerTrainDataset(src_vocab, tgt_vocab, q_train, a_train_input)
    hidden_dim = 128
    embedding_dim = 100
    hidden_dims = [128, 256, 512]
    embedding_dims = [64, 128, 256]
    bidirectional_encoding = True
    directions = [bidirectional_encoding, not bidirectional_encoding]
    direction_dict = {True: "bidirectional", False: "Single"}


    #attention = BahdanauAttention(hidden_dim=h, bidirectional_enc=d)
    #qna_bot = QuestionAnswerer(src_vocab, tgt_vocab, attention, h, e, d)

    for d in directions:
        for h in hidden_dims:
            for e in embedding_dims:
                attention = UniformAttention()
                qna_bot = train(src_vocab=src_vocab, tgt_vocab=tgt_vocab, attention=attention,
                                hidden_dim=h, embedding_dim=e, bidirectional=d,
                                train_dataset=train_dataset, val_dataset=val_dataset, n_epochs=1,
                                batch_size=128, lr=5e-4)
                path = f'{direction_dict[d]}_hidden-{h}_emb-{e}'
                torch.save({'enc_dec': qna_bot.model.state_dict(),
                            'device': qna_bot.device,
                            'src_vocab': qna_bot.src_vocab,
                            'tgt_vocab': qna_bot.tgt_vocab,
                            'i2w': qna_bot.i2w}, "saved_models/" + path)
                sample_input = val_dataset.create_sample_tensor(["what", "is", "<", "operator", "?"], 10)
                print(["what", "is", "<", "operator", "in", "python" ,"?"])
                print(qna_bot.generate_answers(sample_input, 100))
                print(f'Model saved to {path}')


if __name__ == '__main__':
    main()
