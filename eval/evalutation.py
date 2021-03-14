import sacrebleu


def bleu(question_answerer, dataset):
    src, ref = dataset.get_all_examples()
    translated = question_answerer.generate_answers(src)
    return sacrebleu.raw_corpus_bleu(translated, [ref], 0.01).score
