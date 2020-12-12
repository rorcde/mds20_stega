import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup
import random
import re
import pandas as pd


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_train_test_data(non_modified_data, modified_data,
                        test_size=0.3, random_state=0):

    sss = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state)

    train_index, test_index = next(iter(sss.split(non_modified_data)))
    train_non_modified = np.array(non_modified_data)[
        np.array(train_index)].tolist()
    test_non_modified = np.array(non_modified_data)[
        np.array(test_index)].tolist()
    train_index, test_index = next(iter(sss.split(modified_data)))
    train_modified = np.array(modified_data)[np.array(train_index)].tolist()
    test_modified = np.array(modified_data)[np.array(test_index)].tolist()

    return (train_non_modified, train_modified), (test_non_modified, test_modified)


class CustomDataset(Dataset):
    def __init__(self, non_modified_data, modified_data, transform=None):
        self.data = non_modified_data + modified_data
        self.labels = [0] * len(non_modified_data) + [1] * len(modified_data)
        self.data = list(zip(self.data, self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'utt': self.data[idx][0],
            'label': self.data[idx][1]
        }
        if self.transform is not None:
            sample['utt'] = self.transform(sample['utt'])
        return sample


class CustomTransform():
    def __init__(self, tokenizer, max_len=50):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, item):
        item_cp = item
        return torch.LongTensor(self.tokenizer.encode(item_cp, padding='max_length', max_length=self.max_len))[
            :self.max_len]


def train(model, batcher, args):
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * args.n_epoch * len(batcher['train']),
                                                num_training_steps=args.n_epoch * len(batcher['train']))
    log_soft = nn.LogSoftmax(dim=1)
    soft = nn.Softmax(dim=1)
    criterion = nn.NLLLoss()

    print_every = args.print_every
    best_acc = 0.
    phases = ['train', 'dev']
    acc_array = np.zeros((args.n_epoch, 2))
    for epoch in range(1, args.n_epoch + 1):
        print('-' * 25)
        for phase in phases:
            accuracy = {'all': 1, 'true': 0}
            all_pred_probs = list()
            all_target = list()

            loss_ep = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for batch_id, batch in enumerate(batcher[phase]):
                if phase == 'train':
                    scheduler.step()
                utt = batch['utt']
                target = batch['label']
                logits = model(utt.to(device))

                predictions = logits.logits.argmax(1).tolist()
                accuracy['all'] += len(predictions)
                accuracy['true'] += sum([predictions[i] == target.tolist()[i]
                                         for i in range(len(predictions))])
                all_target.extend(target.tolist())
                all_pred_probs.extend(soft(logits.logits)[:, 1].tolist())

                loss = criterion(log_soft(logits.logits), target.to(device))
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                loss_ep += loss.item()

                if batch_id % print_every == 0 and batch_id > 0:
                    phrase = f"Batch {batch_id} | Accuracy {accuracy['true'] / accuracy['all']:.3f} | Loss {loss_ep / (batch_id + 1)}"
                    print(phrase)

            phrase = f"Phase {phase} Epoch {epoch} | Accuracy {accuracy['true'] / accuracy['all']:.3f} | Loss {loss_ep / (batch_id + 1)} | ROC AUC {roc_auc_score(all_target, all_pred_probs)}"
            print(phrase)

            acc_array[epoch - 1, ['train',
                                  'dev'].index(phase)] = accuracy['true'] / accuracy['all']
            if phase != 'train':
                if acc_array[epoch - 1, ['train',
                                         'dev'].index(phase)] > best_acc:
                    best_acc = acc_array[epoch - 1,
                                         ['train', 'dev'].index(phase)]
                    torch.save({'model_state_dict': model.state_dict(),
                                'best_acc': best_acc},
                               args.checkpoint_path)


@torch.no_grad()
def evaluate(model, batcher, args):
    device = torch.device(args.device)
    log_soft = nn.LogSoftmax(dim=1)
    soft = nn.Softmax(dim=1)
    criterion = nn.NLLLoss()
    all_pred_probs = list()
    all_target = list()
    loss_ep = 0

    accuracy = dict()
    accuracy['all'] = 0
    accuracy['true'] = 0

    model.eval()

    for batch_id, batch in enumerate(batcher['dev']):
        utt = batch['utt']
        target = batch['label']
        logits = model(utt.to(device))
        loss = criterion(log_soft(logits.logits), target.to(device))
        loss_ep += loss.item()

        predictions = logits.logits.argmax(1).tolist()
        accuracy['all'] += len(predictions)
        accuracy['true'] += sum([predictions[i] == target.tolist()[i]
                                 for i in range(len(predictions))])
        all_target.extend(target.tolist())
        all_pred_probs.extend(soft(logits.logits)[:, 1].tolist())

    total_accuracy = accuracy['true'] / accuracy['all']
    total_loss = loss_ep / (batch_id + 1)
    roc_auc = roc_auc_score(all_target, all_pred_probs)

    return {'acc': total_accuracy,
            'roc_auc': roc_auc,
            'loss': total_loss}
