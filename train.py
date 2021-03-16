import os
from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.QnABatcher import QnABatcher
from eval.evalutation import rouge, run_evaluation
from models.TitleGenerator import TitleGenerator
from tqdm import tqdm


def train(src_vocab, tgt_vocab, attention, hidden_dim,
          embedding_dim,bidirectional,train_dataset, val_dataset,n_epochs,batch_size, lr=5e-4):
    question_answerer = TitleGenerator(src_vocab, tgt_vocab, attention,
                                       hidden_dim, embedding_dim, bidirectional)

    batcher = QnABatcher(question_answerer.device)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=batcher)

    optimizer = Adam(question_answerer.model.parameters(), lr=lr)

    try:
        for epoch in range(n_epochs):
            losses = []
            rouge_precision_avg = 0
            rouge_recall_avg = 0
            rouge_f1_avg = 0
            sample = '<none>'
            with tqdm(total=len(train_dataset)) as pbar:
                for i, (src_batch, tgt_batch, tgt_eos_batch) in enumerate(train_loader):
                    question_answerer.model.train()

                    # Forward pass
                    scores = question_answerer.model(src_batch, tgt_batch)
                    scores = scores.view(-1, len(tgt_vocab))

                    # Backward pass
                    optimizer.zero_grad()
                    loss = F.cross_entropy(scores, tgt_eos_batch.view(-1), ignore_index=0)
                    loss.backward()
                    optimizer.step()

                    # Update the diagnostics
                    losses.append(loss.item())
                    pbar.set_postfix(loss=(sum(losses))/ len(losses), rouge_precision_avg=rouge_precision_avg,
                                     rouge_recall_avg=rouge_recall_avg,
                                     rouge_f1_avg=rouge_f1_avg, sample=sample)
                    pbar.update(len(src_batch))

                    if i % 100 == 0:
                        question_answerer.model.eval()
                        rouge_precision_avg, rouge_recall_avg, rouge_f1_avg = rouge(question_answerer, val_dataset, "rouge1")
                        sample_input = val_dataset.create_sample_tensor(["how", "do", "i","run", "python"], 100)
                        sample = question_answerer.generate_answers(sample_input, 10)[0]
            if (epoch + 1) % 5 == 0:
                if bidirectional:
                    model_name = f'bidirectional_hidden-{hidden_dim}_emb-{embedding_dim}'
                else:
                    model_name = f'single_hidden-{hidden_dim}_emb-{embedding_dim}'
                run_evaluation(model_name, "validation.csv",question_answerer,val_dataset, epoch, ["rouge1", "rouge"])
    except KeyboardInterrupt:
        pass
    return question_answerer

