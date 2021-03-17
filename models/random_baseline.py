import random


def generate_random_titles(tgt_vocab, max_len, n_samples):
    generated_titles = []
    inv_tgt_vocab = {i:w for w, i in tgt_vocab.items()}
    for _ in range(n_samples):
        title = ""
        for _ in range(max_len):
            i = random.randrange(1, len(tgt_vocab))
            word = inv_tgt_vocab[i]
            title += word + " "
            if word == '<eos>':
                break
        generated_titles += [title]
    return generated_titles
