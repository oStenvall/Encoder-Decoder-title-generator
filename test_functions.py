import pickle
import random

import torch

from dataset.QuestionAnswerTestDataset import QuestionAnswerTestDataset
from dataset.QuestionAnswerTrainDataset import QuestionAnswerTrainDataset, example
from models.Decoder import Decoder
from models.Encoder import Encoder
from models.EncoderDecoder import EncoderDecoder
from models.QuestionAnswerer import QuestionAnswerer
from models.attention_models import BahdanauAttention


def get_example_batches(dataset, batch_size):
    indices = random.sample(range(0, len(dataset)), batch_size)
    batch_src, batch_tgt = example(dataset, indices[0])
    for i in range(1, len(indices), 1):
        index = indices[i]
        src, tgt = example(dataset, index)
        batch_src = torch.cat((batch_src, src))
        batch_tgt = torch.cat((batch_tgt, tgt))
    return batch_src, batch_tgt


def print_encoder_decoder_shape( encoder, decoder,dataset, batch_size=10):

    batch_src, batch_tgt = get_example_batches(dataset, batch_size)
    print("Input batch:    " + str(batch_src.shape))
    print("Target batch:   " + str(batch_tgt.shape))
    src_mask = (batch_src != 0)
    out, hidden = encoder.forward(batch_src)
    print("Encoder output: " + str(out.shape))
    print("Encoder hidden: " + str(hidden.shape))
    output = decoder.forward(encoder_output=out,
                             hidden=hidden,
                             src_mask=src_mask,
                             tgt=batch_tgt)
    print("Decoder output: " + str(output.shape))


def test_encoder_decoder_model(encoder_decoder, dataset, batch_size=10):
    batch_src, batch_tgt = get_example_batches(dataset, batch_size)
    print(batch_tgt.size(1))
    generated, alphas = encoder_decoder.decode(batch_src, batch_tgt.size(1))
    print(generated.shape)
    print(alphas.shape)


def test_QnA_bot(questionAnswerer, dataset, batch_size=10):

    batch_src, batch_tgt = get_example_batches(dataset, batch_size)
    answers = questionAnswerer.generate_answers(batch_src, batch_tgt.size(1))
    for ans in answers:
        print(ans)


def test():
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
    a_train_target = answer_targets[:25000]
    q_val_input = questions[25000:]
    a_val_input = answer_inputs[25000:]
    a_val_tgt = answer_targets[25000:]

    val_dataset = QuestionAnswerTestDataset(src_vocab, q_val_input,a_val_tgt)
    train_dataset = QuestionAnswerTrainDataset(src_vocab, tgt_vocab, q_train, a_train_input, a_train_target)

    hidden_dim = 128
    embedding_dim = 100
    bidirectional_encoding = True
    attention = BahdanauAttention(hidden_dim=hidden_dim, bidirectional_enc=bidirectional_encoding)
    encoder_decoder = EncoderDecoder(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
                                    attention=attention, hidden_dim=hidden_dim, embedding_dim=embedding_dim,
                                    bidirectional=bidirectional_encoding)
    print("Test encoderDecoder model")
    test_encoder_decoder_model(encoder_decoder, train_dataset)
    q_n_a_bot = QuestionAnswerer(src_vocab, tgt_vocab, attention,
                              hidden_dim, embedding_dim, bidirectional_encoding)
    test_QnA_bot(q_n_a_bot, train_dataset, 1)