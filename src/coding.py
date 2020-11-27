import numpy as np


class PerfectBinaryTree(object):
    def __init__(self, words, probabilities):
        super().__init__()
        assert len(probabilities) == len(set(words))
        assert np.log2(len(probabilities)).is_integer()
        assert np.isclose(np.sum(probabilities), 1)

        sorted_idxes = np.argsort(probabilities)[::-1]
        self.words = np.array(words)[sorted_idxes]

    def decode(self, seq):
        assert set(seq).issubset({0, 1})
        assert len(seq) == int(
            np.log2(len(self.words))
        )
        str_seq = ''.join([str(s) for s in seq])

        idx = int(str_seq, 2)
        return self.words[idx]

    def encode(self, word):
        number = np.where(self.words == word)[0][0]
        bin_number = bin(number)[2:]
        if len(bin_number):
            diff = int(
                np.log2(len(self.words)) - len(bin_number)
            )
            bin_number = '0' * diff + bin_number
        seq = [int(i) for i in bin_number]
        return seq




if __name__ == '__main__':
    words = np.array(['a', 'b', 'c', 'd', 'e', 'f','g', 'h'])

    probs = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.3
    ]

    pbt = PerfectBinaryTree(
        words, probs
    )
    t = pbt.decode([0, 1, 1])
    print(pbt.encode(t))