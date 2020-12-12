import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

from encoding import PerfectBinaryTreeWithoutProbs

TOKEN_COUNT = 16
TOKEN_COUNT_LOG = int(np.log2(TOKEN_COUNT))


def init_model(name='gpt2'):
    model = GPT2LMHeadModel.from_pretrained(name, pad_token_id=50256)
    tokenizer = GPT2Tokenizer.from_pretrained(name, pad_token_id=50256)
    model.eval()
    return model, tokenizer


def get_by_idx(seq, idx):
    start_idx = int(TOKEN_COUNT_LOG * idx)
    end_idx = int(TOKEN_COUNT_LOG * (idx + 1))
    return seq[start_idx:end_idx]


def generate(model, init_tokens, seq, tokenizer):
    generate_count = len(seq) % TOKEN_COUNT_LOG
    if generate_count != 0:
        generate_count = TOKEN_COUNT_LOG - generate_count

    seq = seq + [0] * generate_count
    max_idx = len(seq) // TOKEN_COUNT_LOG

    tokens = tokenizer.encode(init_tokens, return_tensors='pt')
    init_len = tokens.shape[1]

    for idx in range(max_idx):
        out = model.generate(
            tokens,
            max_length=1 + idx + init_len,
            num_return_sequences=TOKEN_COUNT,
            num_beams=TOKEN_COUNT
        )

        cur_seq = get_by_idx(seq, idx)

        candidates = out[:, -1]

        pbt = PerfectBinaryTreeWithoutProbs(candidates)
        idx = pbt.decode(cur_seq)
        idx = np.where(candidates == idx)[0][0]

        candidate = candidates[idx].item()

        tokens = torch.cat(
            (tokens, torch.tensor([[candidate]])), axis=-1
        )

    text = tokenizer.decode(tokens[0])
    return text


def decode(model, tokens):
    init_len = 1
    result_bits = []

    for idx in range(tokens.shape[1] - 1):
        index = idx + 1

        out = model.generate(
            tokens[:, :index],
            max_length=index + init_len,
            num_return_sequences=TOKEN_COUNT,
            num_beams=TOKEN_COUNT
        )

        candidates = out[:, -1]

        pbt = PerfectBinaryTreeWithoutProbs(candidates)
        res_idx = np.where(candidates == tokens[:, index])[0][0]

        bits = pbt.encode(candidates[res_idx].item())
        result_bits.extend(bits)

    return result_bits


def _test(bits_per_word, seq_len):
    global TOKEN_COUNT
    global TOKEN_COUNT_LOG
    TOKEN_COUNT = bits_per_word
    TOKEN_COUNT_LOG = int(np.log2(TOKEN_COUNT))

    seq = list(
        np.random.binomial(1, 0.5, seq_len)
    )

    model, tokenizer = init_model()
    init_words = ['Hello', 'Are', 'Four', 'Good', 'World']
    init_word = np.random.choice(init_words, 1)[0]

    text = generate(model, init_word, seq, tokenizer)
    _tokens = tokenizer.encode(text, return_tensors='pt')
    decoded_seq = decode(model, _tokens)

    assert np.array_equal(decoded_seq[:len(seq)], seq)
    print(f'Bits per word: {bit_per_words} text: {text}')


if __name__ == '__main__':
    model, tokenizer = init_model()

    for bit_per_words in [2, 4, 8, 16, 32]:
        _test(16, 70)

    # seq = [
    #     1, 0, 1, 1,
    #     1, 1, 1, 1,
    #     1, 1, 1, 1,
    #     1, 1, 1, 0,
    #     1, 0, 0, 1
    # ]
    # text = generate(model, 'Hello', seq, tokenizer)
    #
    # _tokens = tokenizer.encode(text, return_tensors='pt')
    # decoded_seq = decode(model, _tokens)
    #
    # assert np.array_equal(decoded_seq[:len(seq)], seq)
    #
    # print(text)
