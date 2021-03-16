import pickle
import random
from pathlib import Path

import torch

from dataset.TitleQuestionTestDataset import TitleQuestionTestDataset
from dataset.TitleQuestionTrainDataset import TitleQuestionTrainDataset, example
from eval.evalutation import run_evaluation
from models.Decoder import Decoder
from models.Encoder import Encoder
from models.EncoderDecoder import EncoderDecoder
from models.TitleGenerator import TitleGenerator
from models.attention_models import BahdanauAttention
from models.random_baseline import generate_random_titles


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


def test_random_baseline(tgt_vocab,max_len, n_samples):
    for title in generate_random_titles(tgt_vocab, max_len, n_samples):
        print(title)


def print_example_output(titleGen, dataset):
    test_indecies = [25, 92, 182, 768, 1000, 97, 1, 5, 7]
    for i in test_indecies:
        src_sequence = dataset.get_src_sequence_by_idx(i)
        try:
            eos_index = src_sequence.index("<pad>")
            original_question = ' '.join(src_sequence[:eos_index])
        except ValueError:
            original_question = ' '.join(src_sequence)
        encoder_input, ref_title = dataset[i]
        print(f'Reference title: {original_question}')
        gen_title = titleGen.generate_answers(encoder_input, 10)
        print(f'Generated title: {gen_title}')
        print("Question body")
        print(original_question)
        print('------------------')
        print('---NEXT EXAMPLE---')
        print('------------------')


def load_models_and_print_example_titles_for_best_models(src_vocab,tgt_vocab, test_dataset):
    model_names = ['bidirectional_hidden-64_emb-100',
              'bidirectional_hidden-256_emb-100',
              'single_hidden-64_emb-100','single_hidden-256_emb-100']
    embedding_dim = 100
    hidden_dims = [64, 256,64, 256]
    bidirectional = [True,True,False,False]
    for i in range(len(model_names)):
        model_name = model_names[i]
        h = hidden_dims[i]
        bi = bidirectional[i]
        at = BahdanauAttention(hidden_dim=h, bidirectional_enc=bi)
        titleGen = TitleGenerator(src_vocab, tgt_vocab, at, h, embedding_dim, bi)
        path = Path('saved_models')/ model_name
        titleGen.load(path)
        titleGen.model.eval()
        print_example_output(titleGen, test_dataset)
