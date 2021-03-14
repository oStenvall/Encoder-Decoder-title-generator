import pickle

from dataset.QuestionAnswerTestDataset import QuestionAnswerTestDataset
from dataset.QuestionAnswerTrainDataset import QuestionAnswerTrainDataset
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
    q_train = questions[:25000]
    a_train_input = answer_inputs[:25000]

    q_val_input = questions[25000:]
    a_val_tgt = answer_targets[25000:]

    val_dataset = QuestionAnswerTestDataset(src_vocab, q_val_input,a_val_tgt)
    train_dataset = QuestionAnswerTrainDataset(src_vocab, tgt_vocab, q_train, a_train_input)
    hidden_dim = 128
    embedding_dim = 100
    bidirectional_encoding = True
    attention = BahdanauAttention(hidden_dim=hidden_dim, bidirectional_enc=bidirectional_encoding)
    qna_bot = train(src_vocab=src_vocab, tgt_vocab=tgt_vocab)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
