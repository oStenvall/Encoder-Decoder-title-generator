import math
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import spacy
import pickle

from tqdm import tqdm


class DataConfig:
    def __init__(self):
        self.max_q_len = 10
        self.max_a_len = 100
        self.PAD = '<pad>'
        self.BOS = '<bos>'
        self.EOS = '<eos>'
        self.UNK = '<unk>'
        self.nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner", "lemmatizer"])
        self.pseudowords = ['<pad>', '<bos>', '<eos>', '<unk>']


dataConfig = DataConfig()


def load_data_into_df():
    path_q = Path('../data/Questions.csv')
    path_a = Path('../data/Answers.csv')
    questions_df = pd.read_csv(path_q, encoding='latin-1', nrows=150000)
    answers_df = pd.read_csv(path_a, encoding='latin-1', nrows=150000)
    answers_df = answers_df.drop_duplicates(subset=["ParentId"])
    combined_df = questions_df.join(answers_df.set_index("ParentId"), on="Id", lsuffix='_question', rsuffix='_answer')
    combined_df = combined_df[combined_df['Id_answer'].notna()]
    combined_df = combined_df[combined_df['Body_answer'].notna()]
    return combined_df


def tokenize(sentence):
    soup_sentence = BeautifulSoup(sentence)
    text_sentence = soup_sentence.get_text()
    doc_sentence = dataConfig.nlp(text_sentence)
    return [token.lower_ for token in doc_sentence]


def add_padding(tokens, max_len):
    padding = [dataConfig.PAD] * (max_len - len(tokens))
    return tokens + padding


def extract_question_and_answer(df):
    questions = []
    answers_input = []
    answers_target = []
    with tqdm(total=df.shape[0]) as pbar:
        for i, row in df.iterrows():
            if i % 1000 == 0:
                pbar.update(1000)

            q_tokens = tokenize(row.Title)
            if len(q_tokens) > dataConfig.max_q_len or len(q_tokens) == 0:
                continue

            q_tokens = q_tokens
            if len(q_tokens) < dataConfig.max_q_len:
                q_tokens = add_padding(q_tokens, dataConfig.max_q_len)

            a_tokens_input = tokenize(row.Body_answer)
            a_tokens_target = tokenize(row.Body_answer)
            if len(a_tokens_input) > dataConfig.max_a_len or len(a_tokens_input) == 0:
                continue

            a_tokens_input = [dataConfig.BOS] + a_tokens_input
            a_tokens_target = a_tokens_target #+ [dataConfig.EOS]
            max_a_len_with_pseudo_word = dataConfig.max_a_len + 1
            if len(a_tokens_input) < max_a_len_with_pseudo_word:
                a_tokens_input = add_padding(a_tokens_input, max_a_len_with_pseudo_word)
                #a_tokens_target = add_padding(a_tokens_target, max_a_len_with_pseudo_word)

            assert len(a_tokens_input) == max_a_len_with_pseudo_word
            assert len(q_tokens) == dataConfig.max_q_len

            questions.append(q_tokens)
            answers_input.append(a_tokens_input)
            answers_target.append(a_tokens_target)

    return questions, answers_input, answers_target


def make_vocab(sequences, max_size):
    vocab_with_counter = {}
    index = 4
    for i, token in enumerate(dataConfig.pseudowords):
        vocab_with_counter[token] = {'count': math.inf, 'index': i}
    for sequence in sequences:
        for token in sequence:
            if token not in dataConfig.pseudowords:
                if token not in vocab_with_counter:
                    vocab_with_counter[token] = {'count': 1, 'index': index}
                    index += 1
                else:
                    vocab_with_counter[token]['count'] += 1
    sorted_vocab = sorted(vocab_with_counter.items(), key=lambda x: x[1]['count'], reverse=True)
    vocab = {word: i for i, (word, _) in enumerate(sorted_vocab) if i < max_size}
    return vocab


def save_to_pickle():
    df = load_data_into_df()
    questions, answer_inputs, answer_targets = extract_question_and_answer(df)
    src_vocab = make_vocab(questions, 10000)
    tgt_vocab = make_vocab(answer_inputs, 10000)

    pickle_dict = {"questions": questions,
                   "answer_inputs": answer_inputs,
                   "answer_targets": answer_targets,
                   "src_vocab": src_vocab,
                   "tgt_vocab": tgt_vocab}
    with open('../data/data.p', "wb") as fp:
        pickle.dump(pickle_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    save_to_pickle()
