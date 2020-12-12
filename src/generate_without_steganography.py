import argparse
import torch
from utils import random_seed, get_popular_first_words
from pathlib import Path
import random
import numpy as np
import tqdm
import json
from transformers import AutoTokenizer, AutoModelWithLMHead


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='distilgpt2',
                        help='model checkpoint to use')
    parser.add_argument('--data_path', type=str, default='data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--out_path', type=str, default='experiment/generated.json',
                        help='path to save generated text')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--utterances_to_generate', type=int, default=1000,
                        help='number of unique starting words')
    parser.add_argument('--sentences_per_unique_start', type=int, default=5,
                        help='number of utterances per unique start')
    parser.add_argument('--do_sample', action='store_true',
                        help='use sampling or n ')
    parser.add_argument('--top_k', type=int, default=25,
                        help='top_k')
    parser.add_argument('--top_p', type=float, default=0.6,
                        help='top_p')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='penalty of repeating')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of generating')
    parser.add_argument('--max_length', type=int, default=60,
                        help='maximum tokens in generated utterance')
    parser.add_argument('--min_length', type=int, default=30,
                        help='maximum tokens in generated utterance')
    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    most_common_first_words = get_popular_first_words(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelWithLMHead.from_pretrained(args.model_path,
                                                pad_token_id=tokenizer.eos_token_id
                                                ).to(args.device)
    model.eval()
    print('loaded model')

    soft = torch.nn.Softmax(0)
    generated_strings = list()

    for word_id, word in tqdm.tqdm(enumerate(most_common_first_words)):
        input_ids = tokenizer.encode(word, return_tensors='pt')
        input_ids = input_ids.repeat(
            args.sentences_per_unique_start, 1).to(
            args.device)
        output = model.generate(
            input_ids,
            do_sample=args.do_sample,
            max_length=args.max_length,
            min_length=args.min_length,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
        )
        utt = [tokenizer.decode(i, skip_special_tokens=True) for i in output]
        generated_strings.extend(utt)

    json.dump(generated_strings, Path(args.out_path).open('w'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
