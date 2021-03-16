import pickle

import torch

from dataset.TitleQuestionTestDataset import TitleQuestionTestDataset
from dataset.TitleQuestionTrainDataset import TitleQuestionTrainDataset
from eval.evalutation import run_evaluation, evaluate_random_baseline
from models.attention_models import BahdanauAttention
from test_functions import load_models_and_print_example_titles_for_best_models
from train import train


def main():
    file = open('data/question_title_body_1000_words.p', "rb")
    data_and_vocab = pickle.load(file)
    src_vocab = data_and_vocab["src_vocab"]
    tgt_vocab = data_and_vocab["tgt_vocab"]
    print(len(src_vocab))
    print(len(tgt_vocab))
    assert len(src_vocab) == len(tgt_vocab)
    q_titles_bos = data_and_vocab["q_titles_bos"]
    q_titles_eos = data_and_vocab["q_titles_eos"]
    q_bodies = data_and_vocab["q_bodies"]
    q_titles_ref = data_and_vocab["q_titles_ref"]

    q_bodies_train = q_bodies[0:9000]
    q_titles_bos_train = q_titles_bos[0:9000]
    q_titles_eos_train = q_titles_eos[0:9000]

    q_titles_ref_val = q_titles_ref[9000:10000]
    q_bodies_val = q_bodies[9000:10000]

    q_titles_ref_test = q_titles_ref[10000:12328]
    q_bodies_test = q_bodies[10000:12328]
    #ref_a_val = ref_a[25000:]

    test_dataset = TitleQuestionTestDataset(src_vocab, q_bodies_test, q_titles_ref_test)
    val_dataset = TitleQuestionTestDataset(src_vocab, q_bodies_val, q_titles_ref_val)
    train_dataset = TitleQuestionTrainDataset(src_vocab, tgt_vocab, q_bodies_train, q_titles_bos_train, q_titles_eos_train)
    #attention = UniformAttention()
    #enc = Encoder(vocab_size=len(src_vocab))
    #attention = BahdanauAttention()
    #dec = Decoder(vocab_size=len(tgt_vocab),attention=attention)
    #print_encoder_decoder_shape(encoder=enc,decoder=dec,dataset=train_dataset,batch_size=10)

    hidden_dims = [64, 128, 256]
    embedding_dims = [100]
    bidirectional_encoding = True
    directions = [not bidirectional_encoding, bidirectional_encoding]
    direction_dict = {True: "bidirectional", False: "single"}
    #load_models_and_print_example_titles_for_best_models(src_vocab,tgt_vocab,test_dataset)
    #load_models_and_calculate_rouge(src_vocab,tgt_vocab, test_dataset)
    #attention = BahdanauAttention(hidden_dim=h, bidirectional_enc=d)
    #qna_bot = QuestionAnswerer(src_vocab, tgt_vocab, attention, h, e, d)

    for d in directions:
         for h in hidden_dims:
             for e in embedding_dims:
                attention = BahdanauAttention(hidden_dim=h, bidirectional_enc=d)
                qna_bot = train(src_vocab=src_vocab, tgt_vocab=tgt_vocab, attention=attention,
                                hidden_dim=h, embedding_dim=e, bidirectional=d,
                                train_dataset=train_dataset, val_dataset=val_dataset, n_epochs=25,
                                batch_size=128, lr=5e-4)
                model_name = f'{direction_dict[d]}_hidden-{h}_emb-{e}'
                torch.save({'enc_dec': qna_bot.model.state_dict(),
                            'device': qna_bot.device,
                            'src_vocab': qna_bot.src_vocab,
                            'tgt_vocab': qna_bot.tgt_vocab,
                            'i2w': qna_bot.i2w}, "saved_models/" + model_name)
                sample_input = val_dataset.create_sample_tensor(["what", "is", "<", "operator", "?"], 10)
                print(["what", "is", "<", "operator", "in", "python" ,"?"])
                print(qna_bot.generate_answers(sample_input, 10))
                print(f'Model saved to {model_name}')
                #run_evaluation(model_name,"test.csv", qna_bot, test_dataset, 25)


if __name__ == '__main__':
    main()
