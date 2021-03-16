import math
import pdb

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import spacy
import pickle

from tqdm import tqdm


class DataConfig:
    def __init__(self):
        self.max_q_title_len = 10
        self.max_q_body_len = 100
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
    path_t = Path('../data/Tags.csv')
    questions_df = pd.read_csv(path_q, encoding='latin-1', nrows=150000)
    answers_df = pd.read_csv(path_a, encoding='latin-1', nrows=150000)
    tags_df = pd.read_csv(path_t, encoding='latin-1', nrows=150000)
    answers_df = answers_df.drop_duplicates(subset=["ParentId"])
    combined_df = questions_df.join(answers_df.set_index("ParentId"), on="Id", lsuffix='_question', rsuffix='_answer')
    #tags_grouped = tags_df.groupby('Id', as_index=False).agg(lambda x: x.tolist())
    tags_df = tags_df[tags_df['Tag'].str.contains("python", na=False)]
    combined_df = combined_df.join(tags_df.set_index('Id'), on="Id_question")
    combined_df = combined_df[combined_df["Tag"].notna()]
    combined_df = combined_df[combined_df['Id_answer'].notna()]
    combined_df = combined_df[combined_df['Body_question'].notna()]
    return combined_df


def tokenize(sentence):
    soup_sentence = BeautifulSoup(sentence)
    text_sentence = soup_sentence.get_text()
    doc_sentence = dataConfig.nlp(text_sentence)
    return [token.lower_ for token in doc_sentence]


def add_padding(tokens, max_len):
    if len(tokens) < max_len:
        padding = [dataConfig.PAD] * (max_len - len(tokens))
    else:
        padding = []
    return tokens + padding


def extract_question_and_answer(df):
    q_titles_eos = []
    q_titles_bos = []
    q_bodies = []
    q_titles_ref = []
    with tqdm(total=df.shape[0]) as pbar:
        for i, row in df.iterrows():
            if i % 1000 == 0:
                pbar.update(1000)

            try:
                q_title = tokenize(row.Title)
            except UnicodeEncodeError:
                continue
            q_title

            if len(q_title) > dataConfig.max_q_title_len or len(q_title) == 0:
                continue

            q_title_bos = [dataConfig.BOS]
            q_title_bos += q_title
            q_title_bos += [dataConfig.EOS]
            q_title_eos = q_title
            q_title_eos += [dataConfig.EOS]
            #pdb.set_trace()
            if len(q_title_bos) < dataConfig.max_q_title_len + 2:
                q_title_bos = add_padding(q_title_bos, dataConfig.max_q_title_len + 2)
            if len(q_title_eos) < dataConfig.max_q_title_len + 2:
                q_title_eos = add_padding(q_title_eos, dataConfig.max_q_title_len + 2)
            #pdb.set_trace()
            try:
                q_body = tokenize(row.Body_question)
            except UnicodeEncodeError:
                continue

            if len(q_body) > dataConfig.max_q_body_len or len(q_body) == 0:
                continue

            q_body = q_body
            if len(q_body) < dataConfig.max_q_body_len:
                q_body = add_padding(q_body, dataConfig.max_q_body_len)


            assert len(q_body) == dataConfig.max_q_body_len
            assert len(q_title_eos) == dataConfig.max_q_title_len + 2
            assert len(q_title_bos) == dataConfig.max_q_title_len + 2

            q_titles_bos.append(q_title_bos)
            q_titles_eos.append(q_title_eos)
            q_bodies.append(q_body)
            q_titles_ref.append(q_title)

    return q_titles_bos, q_titles_eos, q_bodies, q_titles_ref


def make_vocab(sequences, max_size):
    vocab_with_counter = {}
    index = 4
    for i, token in enumerate(dataConfig.pseudowords):
        vocab_with_counter[token] = {'count': math.inf, 'index': i}
    for sequence in sequences:
        for token in sequence:
            if token not in dataConfig.pseudowords:
                if type(token) == list:
                    pdb.set_trace()
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
    q_titles_bos, q_titles_eos, q_bodies, q_titles_ref = extract_question_and_answer(df)
    src_vocab = make_vocab(q_bodies, 1000)
    tgt_vocab = make_vocab(q_titles_bos, 1000)

    pickle_dict = {"q_titles_bos": q_titles_bos,
                   "q_titles_eos": q_titles_eos,
                   "q_bodies": q_bodies,
                   "q_titles_ref": q_titles_ref,
                   "src_vocab": src_vocab,
                   "tgt_vocab": tgt_vocab}
    with open('../data/question_title_body_1000_words.p', "wb") as fp:
        pickle.dump(pickle_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save_to_pickle()
