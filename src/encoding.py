import numpy as np


class PerfectBinaryTreeWithoutProbs(object):
    def __init__(self, words):
        super().__init__()
        assert np.log2(len(words)).is_integer()
        self.words = np.array(words)

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


# Huffman Encoding
# Tree-Node Type
class Node:
    def __init__(self, freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq

    def isLeft(self):
        return self.father.left == self


# create nodes
def createNodes(freqs):
    return [Node(freq) for freq in freqs]


# create Huffman Tree
def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item: item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]


# Huffman encoding
def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes


if __name__ == '__main__':
    words = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

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

