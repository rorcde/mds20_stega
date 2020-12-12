import json
from pathlib import Path
import torch
import math
import numpy as np
import time
from data import Corpus
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit
import random
import re
from model import RNNModel, TransformerModel

import logging
logging.basicConfig(level=logging.INFO, filename='logs.log')


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_popular_first_words(args):
    corpus = Corpus(args.data_path)
    ntokens = len(corpus.dictionary)
    idx2word = corpus.dictionary.idx2word
    most_common_first_words_ids = [i[0] for i in Counter(corpus.train.tolist()).most_common()
                                   if idx2word[i[0]][0].isupper()][:args.utterances_to_generate]
    return[corpus.dictionary.idx2word[i] for i in most_common_first_words_ids]


def create_private_code(length=100):
    return torch.randint(0, 2, (length,))


def get_by_idx(seq, idx, TOKEN_COUNT_LOG):
    start_idx = int(TOKEN_COUNT_LOG * idx)
    end_idx = int(TOKEN_COUNT_LOG * (idx + 1))
    return seq[start_idx:end_idx]


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(args, model, data_source, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval()
    total_loss = 0.
#     ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(args, data_source, i)
            if args.model == 'Transformer':
                output = model(data.to(device))
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data.to(device), hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output,
                                                targets.to(device)).item()
    return total_loss / (len(data_source) - 1)


def train_epoch(args, model, optimizer, train_data,
                criterion, epoch, scheduler=None):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.train()
    total_loss = 0.
    start_time = time.time()
#     ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(args, train_data, i)
        if args.model == 'Transformer':
            output = model(data.to(device))
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data.to(device), hidden)
        loss = criterion(output, targets.to(device))
        loss.backward()
        if scheduler is not None:
            scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            verbose_string = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | ' 'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))
            print(verbose_string)
            logging.info(verbose_string)
            total_loss = 0
            start_time = time.time()


def train(args, model, optimizer, train_data, val_data, scheduler=None):
    criterion = nn.NLLLoss()
    best_val_loss = None
    for epoch_id, epoch in enumerate(range(1, args.epochs + 1)):
        print('=' * 50)
        logging.info('=' * 50)
        verbose_string = f"Epoch {epoch_id + 1} / {args.epochs} starts"
        print(verbose_string)
        logging.info(verbose_string)
        train_epoch(
            args,
            model,
            optimizer,
            train_data,
            criterion,
            epoch,
            scheduler)
        val_loss = evaluate(args, model, val_data, criterion)
        print('=' * 50)
        logging.info('=' * 50)
        verbose_string = '| end of epoch {:3d} |valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            epoch, val_loss, math.exp(val_loss))
        print(verbose_string)
        logging.info(verbose_string)

        if not best_val_loss or val_loss < best_val_loss:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss},
                       args.save)
            print('model saved')
            logging.info('model saved')
            best_val_loss = val_loss
