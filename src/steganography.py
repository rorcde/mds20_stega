import argparse
import torch
import data
from utils import *
from model import RNNModel, TransformerModel
from collections import Counter
import random
import tqdm
from encoding import *

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
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='model checkpoint to use')
    #
    parser.add_argument('--outf', type=str, default='generated.txt',
                        help='output file for generated text')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--len_of_generation', type=int, default='40',
                        help='number of words to generate')
    parser.add_argument('--bit_num', type=int, default=1,
                        help='number of words to generate')
    parser.add_argument('--utterances_to_generate', type=int, default=100,
                        help='number of utterances to generate')
    #paths
    parser.add_argument('--bit_stream_path', type=str, default='data/experiment/bit_stream.txt', help='path to private message')
    parser.add_argument('--save_path', type=str, default='data/experiment/',
                        help='path to save generated messages(from space S in paper notation) and extracted')
    
    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")
   
    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    word2idx = corpus.dictionary.word2idx
    idx2word = corpus.dictionary.idx2word
    args.vocab_size = len(word2idx)
    print('loaded dictionary')
    
    if args.model == 'Transformer':
        model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    print('loaded model')
 
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    
    #get as starting words only most common starting word 
    #from data corpus(heuristics from baseline)  
    most_common_first_words_ids = [i[0] for i in Counter(corpus.train.tolist()).most_common() 
                                   if idx2word[i[0]][0].isupper()][:200]
#     most_common_first_words = [corpus.dictionary.idx2word[i] 
#                                for i in most_common_first_words_ids]

    #private message(binary code)
    bit_stream = open(args.bit_stream_path, 'r').readline()
    outfile = open(args.save_path + 'generated' + str(args.bit_num) + '_bit.txt', 'w')
    bitfile = open(args.save_path + 'bitfile_' + str(args.bit_num) + '_bit.txt', 'w')
    bit_index = random.randint(0, len(word2idx))
    soft = torch.nn.Softmax(0)
    
    for uter_id, uter in tqdm.tqdm(enumerate(range(args.utterances_to_generate))):
#         with torch.no_grad():  # no tracking history
            input_ = torch.LongTensor([random.choice(
                most_common_first_words_ids)]).unsqueeze(0).to(device)
            if not is_transformer_model:
                hidden = model.init_hidden(1)
                
            output, hidden = model(input_, hidden)
            gen = np.random.choice(len(corpus.dictionary), 1, 
                                   p = np.array(soft(output.reshape(-1)).tolist()) /
                                   sum(soft(output.reshape(-1)).tolist()))[0]
            gen_res = list()
            gen_res.append(idx2word[gen])
            bit = ""
            for word_id, word in enumerate(range(args.len_of_generation -2)):
                if is_transformer_model:
                    assert NotImplementedError
                else:
                    output, hidden = model(input_, hidden)
                p = output.reshape(-1)
                sorted_, indices = torch.sort(p, descending = True)
                words_prob = [(j, i) for i,j in 
                             zip(sorted_[:2**int(args.bit_num)].tolist(),
                                 indices[:2**int(args.bit_num)].tolist())]
                    
                
                nodes = createNodes([item[1] for item in words_prob])
                root = createHuffmanTree(nodes)
                codes = huffmanEncoding(nodes, root)
                
                for i in range(2**int(args.bit_num)):
                    if bit_stream[bit_index:bit_index+i+1] in codes:
                        code_index = codes.index(bit_stream[bit_index:bit_index+i+1])
                        gen = words_prob[code_index][0]
                        test_data = np.int32(gen)
                        gen_res.append(idx2word[gen])
                        if idx2word[gen] in ['\n', '', "<eos>"]:
                            break
                        bit += bit_stream[bit_index: bit_index+i+1]
                        bit_index = bit_index+i+1
                        break
                        
            gen_sen = ' '.join([word for word in gen_res if word not in ["\n", "", "<eos>"]])
            outfile.write(gen_sen+"\n")
            bitfile.write(bit)
                
if __name__ == '__main__':
    args = parse_arguments()
    main(args)