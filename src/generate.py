import argparse
import torch
import data
from utils import *
from model import RNNModel, TransformerModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    # model params
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    #
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='generated.txt',
                        help='output file for generated text')
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
    args = parser.parse_args()

    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    print('loaded dictionary')
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

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('loaded model')

    is_transformer_model = hasattr(
        model, 'model_type') and model.model_type == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    with open(args.outf, 'w') as outf:
        with torch.no_grad():  # no tracking history
            for i in range(args.words):
                if is_transformer_model:
                    output = model(input, False)
                    word_weights = output[-1].squeeze().div(
                        args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    input = torch.cat([input, word_tensor], 0)
                else:
                    output, hidden = model(input, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    input.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
