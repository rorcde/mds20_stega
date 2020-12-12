from pathlib import Path
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from utils_discriminator import (random_seed,
                                 get_train_test_data,
                                 CustomDataset,
                                 CustomTransform,
                                 train,
                                 evaluate)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--transformers_path',
        type=str,
        default='roberta-base/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_folder', type=str, default='results')
    parser.add_argument(
        '--non_modified_data',
        type=str,
        default='experiment/generated.json')
    parser.add_argument('--result_path', type=str, default='results_stat.json')
    parser.add_argument('--checkpoint_path', type=str, default='trained.pt')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device)
    if args.seed is not None:
        random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.transformers_path)
    non_modified_data = json.load(Path(args.non_modified_data).open('r'))

    result = dict()
    paths = Path(args.data_folder).glob('*.json')
    for path in paths:
        print('=' * 50)
        method, bpt = str(path).split('/')[2].split('.')[0].split('_')
        bpt = int(bpt)
        print(f"method: {method}, beats_per_token: {bpt}")
        if bpt not in result.keys():
            result[bpt] = dict()
        result[bpt][method] = {'acc': list(), 'roc_auc': list()}

        for j in range(args.n_splits):
            print(f"{j + 1} SPLIT OUT OF {args.n_splits}")
            seed = random.randint(0, 10e6)

            modified_data = json.load(Path(path).open('r'))
            train_data, test_data = get_train_test_data(non_modified_data,
                                                        modified_data,
                                                        test_size=args.test_size,
                                                        random_state=seed)
            transform = CustomTransform(tokenizer, max_len=100)
            train_dataset = CustomDataset(
                train_data[0], train_data[1], transform=transform)
            test_dataset = CustomDataset(
                test_data[0], test_data[1], transform=transform)
            batcher = {'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
                       'dev': DataLoader(test_dataset, batch_size=args.batch_size)}

            config = AutoConfig.from_pretrained(
                args.transformers_path, num_labels=2)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.transformers_path, config=config).to(device)

            train(model, batcher, args)
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            current_res = evaluate(model, batcher, args)
            result[bpt][method]['acc'].append(current_res['acc'])
            result[bpt][method]['roc_auc'].append(current_res['roc_auc'])

            json.dump(result, Path(args.result_path).open('w'))
            del model, checkpoint, train_data, test_data, train_dataset, test_dataset, batcher


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
