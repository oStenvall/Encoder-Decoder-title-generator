import csv
import os
from pathlib import Path

from rouge_score import rouge_scorer

from models.TitleGenerator import TitleGenerator
from models.attention_models import BahdanauAttention
from models.random_baseline import generate_random_titles


def run_evaluation(model_name, file, question_answerer, dataset, epoch, rouge_types, tgt_vocab=None):
    for rouge_type in rouge_types:
        if model_name == "random":
            rouge_precision_avg, rouge_recall_avg, rouge_f1_avg = rouge(question_answerer, dataset, rouge_type, random=True, tgt_vocab=tgt_vocab)
        else:
            rouge_precision_avg, rouge_recall_avg, rouge_f1_avg = rouge(question_answerer, dataset, rouge_type)
        rouge_dict = {"epoch": str(epoch), "model": model_name,"precision_avg": str(rouge_precision_avg), "recall_avg": str(rouge_recall_avg), "f1_avg": str(rouge_f1_avg)}
        path = Path(f"eval_result_ban/{file}")
        file_exists = os.path.isfile(path)
        with open(path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=rouge_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(rouge_dict)
            csv_file.close()


def rouge(question_answerer, dataset, rouge_type, random=False, tgt_vocab=None):
    if not random:
        src, ref_batch = dataset.get_all_examples()
        answers = question_answerer.generate_answers(src, 10)
    else:
        _, ref_batch = dataset.get_all_examples()
        answers = generate_random_titles(tgt_vocab, 10, len(ref_batch))
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)

    rouge_precision_avg = 0
    rouge_recall_avg = 0
    rouge_f1_avg = 0
    for i in range(len(answers)):
        if len(answers[i]) < len(ref_batch[i]):
            ans = answers[i][0]
            ref = ref_batch[i]
            scores = scorer.score(ref, ans)
            rouge_precision_avg += scores[rouge_type].precision
            rouge_recall_avg += scores[rouge_type].recall
            rouge_f1_avg += scores[rouge_type].fmeasure
    rouge_precision_avg = rouge_precision_avg / len(ref_batch)
    rouge_recall_avg = rouge_recall_avg / len(ref_batch)
    rouge_f1_avg = rouge_f1_avg / len(ref_batch)
    return rouge_precision_avg, rouge_recall_avg, rouge_f1_avg


def load_models_and_calculate_rouge(src_vocab,tgt_vocab, test_dataset):
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    model_names = ['bidirectional_hidden-64_emb-100', 'bidirectional_hidden-128_emb-100',
              'bidirectional_hidden-256_emb-100',
              'single_hidden-64_emb-100','Single_hidden-128_emb-100','single_hidden-256_emb-100']
    embedding_dim = 100
    hidden_dims = [64, 128, 256,64, 128, 256]
    bidirectional = [True,True,True,False,False,False]
    for i in range(len(model_names)):
        model_name = model_names[i]
        h = hidden_dims[i]
        bi = bidirectional[i]
        at = BahdanauAttention(hidden_dim=h, bidirectional_enc=bi)
        titleGen = TitleGenerator(src_vocab, tgt_vocab, at, h, embedding_dim, bi)
        path = Path('saved_models')/ model_name
        titleGen.load(path)
        titleGen.model.eval()
        for rouge_type in rouge_types:
            filename = rouge_type + "test.csv"
            run_evaluation(model_name, filename, titleGen,
                           test_dataset, 25, rouge_type)


def evaluate_random_baseline(tgt_vocab,test_dataset):
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    for rouge_type in rouge_types:
        filename = rouge_type + "test.csv"
        run_evaluation("random", filename, None, test_dataset, '-', rouge_type, tgt_vocab)

