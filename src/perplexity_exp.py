import argparse
import torch
from utils import random_seed, get_popular_first_words, create_private_code, get_by_idx
from encoding import createNodes, createHuffmanTree, huffmanEncoding, PerfectBinaryTreeWithoutProbs
from pathlib import Path
import random
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelWithLMHead

def vlc(args, model, tokenizer, start_word, private_code, m = 10):
    private_code = ''.join([str(i) for i in private_code.tolist()])
    soft = torch.nn.Softmax(-1)
    input_ids = tokenizer.encode(start_word, 
                                 return_tensors='pt', 
                                 add_special_tokens=True).to(args.device)
    input_prepared = model.prepare_inputs_for_generation(input_ids, return_dict=True)
    prob = list()
    number_of_codes_used = 0
    while len(private_code) > np.log2(m):
        outputs = model(**input_prepared, return_dict=True)
        outputs.logits[:, -1, :][0][tokenizer.eos_token_id]  = -1e4
        sorted_, indices = torch.sort(soft(outputs.logits[:, -1, :]), descending = True)
        words_prob = [(j, i) for i,j in zip(sorted_[0][:m].tolist(), indices[0][:m].tolist())]
        nodes = createNodes([item[1] for item in words_prob])
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes, root)
        for i in codes:
            if private_code.startswith(i):
                private_code = private_code[len(i):]
                idx = codes.index(i)
                break
        
        input_prepared['input_ids'] = torch.cat((input_prepared['input_ids'], 
                                                torch.Tensor([words_prob[idx][0]]).unsqueeze(0).long().to(args.device)), 1)
        prob.append(np.log(words_prob[idx][1]))
        number_of_codes_used +=1 
    
    perplexity = 2 ** (-np.mean(prob))
    return perplexity, input_prepared['input_ids'], number_of_codes_used

def flc(args,model, tokenizer, start_word, private_code, m = 10):
    private_code = private_code.tolist()
    TOKEN_COUNT_LOG = int(np.log2(m))
    if TOKEN_COUNT_LOG > 0:
        generate_count = len(private_code) % TOKEN_COUNT_LOG
    else:
        generate_count = 0
    if generate_count != 0:
        generate_count = TOKEN_COUNT_LOG - generate_count
    private_code = private_code + [0] * generate_count
    
    if TOKEN_COUNT_LOG > 0:
        max_idx = len(private_code) // TOKEN_COUNT_LOG
    else:
        max_idx = len(private_code) // m
    soft = torch.nn.Softmax(-1)
    
    input_ids = tokenizer.encode(start_word, 
                                 return_tensors='pt', 
                                 add_special_tokens=True).to(args.device)
    input_prepared = model.prepare_inputs_for_generation(input_ids, return_dict=True)
    prob = list()
    
    for i_id in range(max_idx):
        outputs = model(**input_prepared, return_dict=True)
        outputs.logits[:, -1, :][0][tokenizer.eos_token_id]  = -1e4
        sorted_, indices = torch.sort(soft(outputs.logits[:, -1, :]), descending = True)
        words_prob = [(j, i) for i,j in zip(sorted_[0][:m].tolist(), indices[0][:m].tolist())]
        
        cur_seq = get_by_idx(private_code, i_id, TOKEN_COUNT_LOG)
        pbt = PerfectBinaryTreeWithoutProbs([item[0] for item in words_prob])
        idx = pbt.decode(cur_seq)
        idx = np.where([item[0] for item in words_prob] == idx)[0][0]
        
        input_prepared['input_ids'] = torch.cat((input_prepared['input_ids'], 
                                                torch.Tensor([words_prob[idx][0]]).unsqueeze(0).long().to(args.device)), 1)
        prob.append(np.log(words_prob[idx][1]))
    
    perplexity = 2 ** (-np.mean(prob))
        
    return perplexity, input_prepared['input_ids']
        

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='distilgpt2', 
                        help='model checkpoint to use')
    parser.add_argument('--data_path', type=str, default='data/wikitext-2',
                    help='location of the data corpus')
    parser.add_argument('--out_folder', type=str, default='experiment/',
                        help='folder path to save generated text and metrics')
    parser.add_argument('--encoding_type', type=str, default='FLC',
                        help='output file for generated text', 
                        choices=['FLC', 'VLC', 'no_steganographic'])
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--len_of_private_code', type=int, default=100,
                        help='number of bits to hide')
    parser.add_argument('--utterances_to_generate', type=int, default=1000,
                        help='number of utterances to generate per each encoding type')
    parser.add_argument('--m', '--beats_per_token', type=int, default=8,
                        help='number of available codes on each step')
    
    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    most_common_first_words = get_popular_first_words(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)  
    model = AutoModelWithLMHead.from_pretrained(args.model_path).to(args.device)
    model.eval()
    print('loaded model')
    
    Path(args.out_folder).mkdir(parents=True, exist_ok=True)
    Path(args.out_folder, 'texts').mkdir(parents=True, exist_ok=True)
    Path(args.out_folder, 'metrics').mkdir(parents=True, exist_ok=True)
    
    soft = torch.nn.Softmax(0)
    generated_strings = list()
    perplexities = list()
    number_of_codes_used = list()
    
    for word_id, word in enumerate(most_common_first_words):
        private_code = create_private_code(args.len_of_private_code)
        if args.encoding_type == 'FLC':
            perp, generated = flc(args, model, tokenizer,word, private_code, args.m)
        elif args.encoding_type == 'VLC':
            perp, generated, n_of_codes = vlc(args, model, tokenizer, word, private_code, args.m)
            number_of_codes_used.append(n_of_codes)
        else:
             raise NotImplementedError
        
        generated_strings.append(tokenizer.decode(generated[0]))
        perplexities.append(perp)
        
    metric = {'perplexity': np.mean(perplexities)}
    if args.encoding_type == 'VLC':
        metric['beats_per_token'] = len(most_common_first_words) * args.len_of_private_code / np.sum(number_of_codes_used)
    
    json.dump(metric, Path(args.out_folder, 'metrics', f"{args.encoding_type}_{args.m}.json").open('w'))
    json.dump(generated_strings, Path(args.out_folder, 'texts', f"{args.encoding_type}_{args.m}.json").open('w'))
    
                
if __name__ == '__main__':
    args = parse_arguments()
    main(args)