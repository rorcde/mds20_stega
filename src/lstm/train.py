from utils import *
from model import RNNModel, TransformerModel
import data
import argparse
import torch
from torch import optim
import logging
logging.basicConfig(level=logging.INFO, filename='logs.log')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')

    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    corpus = data.Corpus(args.data)
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.batch_size)
    test_data = batchify(corpus.test, args.batch_size)
    print('loaded data')
    print(f'number of unique tokens: {len(corpus.dictionary)}')

    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        model = TransformerModel(
            ntokens,
            args.emsize,
            args.nhead,
            args.nhid,
            args.nlayers,
            args.dropout).to(device)
    else:
        model = RNNModel(
            args.model,
            ntokens,
            args.emsize,
            args.nhid,
            args.nlayers,
            args.dropout,
            args.tied).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.001,
                                                    steps_per_epoch=len(list(range(0,
                                                                                   train_data.size(
                                                                                       0) - 1,
                                                                                   args.bptt))),
                                                    epochs=args.epochs,
                                                    anneal_strategy='linear')
    print('initialized model and optimizer')
    train(args, model, optimizer, train_data, val_data, scheduler)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
