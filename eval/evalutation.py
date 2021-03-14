import pdb
from rouge_score import rouge_scorer

import sacrebleu
from rouge import Rouge


def bleu(question_answerer, dataset):
    src, ref = dataset.get_all_examples()
    translated = question_answerer.generate_answers(src, 100)
    return sacrebleu.raw_corpus_bleu(translated, [ref], 0.01).score


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
