import csv
import os
import pdb
from pathlib import Path

from rouge_score import rouge_scorer

import sacrebleu
from rouge import Rouge


def bleu(question_answerer, dataset):
    src, ref = dataset.get_all_examples()
    translated = question_answerer.generate_answers(src, 100)
    return sacrebleu.raw_corpus_bleu(translated, [ref], 0.01).score


def run_evaluation(model, file, question_answerer, dataset, epoch):
    rouge_precision_avg, rouge_recall_avg, rouge_f1_avg = rouge(question_answerer, dataset)
    rouge_dict = {"epoch": str(epoch), "model": model,"precision_avg": str(rouge_precision_avg), "recall_avg": str(rouge_recall_avg), "f1_avg": str(rouge_f1_avg)}
    path = Path(f"eval_results/{file}")
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=rouge_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(rouge_dict)
        csv_file.close()


def rouge(question_answerer, dataset):
    src, ref_batch = dataset.get_all_examples()
    #id2w_src = {i: w for w, i in question_answerer.src_vocab}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

    answers = question_answerer.generate_answers(src, 100)

    rouge_precision_avg = 0
    rouge_recall_avg = 0
    rouge_f1_avg = 0
    for i in range(len(answers)):
        if len(answers[i]) < len(ref_batch[i]):
            ans = answers[i][0]
            ref = ref_batch[i]
            scores = scorer.score(ref, ans)
            rouge_precision_avg += scores['rouge1'].precision
            rouge_recall_avg += scores['rouge1'].recall
            rouge_f1_avg += scores['rouge1'].fmeasure
    rouge_precision_avg = rouge_precision_avg / len(src)
    rouge_recall_avg = rouge_recall_avg / len(src)
    rouge_f1_avg = rouge_f1_avg / len(src)
    return rouge_precision_avg, rouge_recall_avg, rouge_f1_avg
