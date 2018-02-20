"""

An Example for
Hidden Markov Models and the Viterbi Algorithm
Implemented in Python

References:
- https://web.stanford.edu/~jurafsky/slp3/9.pdf

Supported Corpuses:
- treebank
- brown

"""

import argparse

import hashlib

import numpy as np

import nltk


def hash(s):
    """ Deterministic string hashing. """
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16)


# Data

def parse_split_str(split_str):
    splits = list(map(int, split_str.split(',')))
    assert sum(splits) == 100, "Sum of splits must equal 100."
    assert len(splits) == 3, "Must have train/dev/test splits."
    return splits


def split(dataset, splits):
    if isinstance(splits, str):
        splits = parse_split_str(splits)

    dataset_size = len(dataset)
    offset = 0

    datasets = []

    for s in splits:
        pct = s / 100
        until = min(int(offset + pct * dataset_size), dataset_size)
        datasets.append(dataset[offset:until])
        offset = until

    return datasets


# HMM

START = 'start'

class HMM(object):

    def __init__(self):
        self.__word_vocab_size = 2
        self.tags = [0, 1]
        self.__tag_vocab_size = len(self.tags)
        self.__w_given_t_table = {
            (0, 0): 0.1,
            (0, 1): 0.9,
            (1, 0): 0.6,
            (1, 1): 0.4
        }

        self.__t1_given_t0_table = {
            (0, START): 0.5,
            (1, START): 0.5,
            (0, 0): 0.3,
            (0, 1): 0.7,
            (1, 0): 0.55,
            (1, 1): 0.45
        }

    def __t1_given_t0(self, tag1, tag0):
        t1 = hash(tag1) % self.__tag_vocab_size
        t0 = hash(tag0) % self.__tag_vocab_size
        return self.__t1_given_t0_table[(t1, t0)]

    def __w_given_t(self, word, tag):
        w = hash(word) % self.__word_vocab_size
        t = hash(tag) % self.__tag_vocab_size
        return self.__w_given_t_table[(w, t)]

    def __compute_trellis(self, sentence):

        trellis = np.empty((len(sentence), self.__tag_vocab_size), dtype=np.float32)

        w = hash(sentence[0])  % self.__word_vocab_size
        for j, t in enumerate(self.tags):
            trellis[0, j] = self.__t1_given_t0_table[(t, START)] * \
                self.__w_given_t_table[(w, t)]

        for i in range(1, len(sentence)):
            w = hash(sentence[i])  % self.__word_vocab_size
            for j, t in enumerate(self.tags):
                cumprob = 0
                for jprev, tprev in enumerate(self.tags):
                    cumprob += trellis[i-1, jprev] * \
                        self.__t1_given_t0_table[(t, tprev)] * \
                        self.__w_given_t_table[(w, t)]
                trellis[i, j] = cumprob

        return trellis

    def __log_likelihood_1(self, sentence, tags):
        func = np.vectorize(lambda w, t: self.__w_given_t(w, t))
        probabilities = func(sentence, tags)
        return np.sum(np.log(probabilities))

    def __log_likelihood_2(self, sentence):
        """ Forward Algorithm """
        trellis = self.__compute_trellis(sentence)

        return np.log(np.sum(trellis[-1]))

    def log_likelihood(self, sentence, tags=None):
        if tags is not None:
            return self.__log_likelihood_1(sentence, tags)
        else:
            return self.__log_likelihood_2(sentence)

    def decode(self, sentence):
        """ Viterbi Algorithm """

        tags = []

        trellis = np.empty((len(sentence), self.__tag_vocab_size), dtype=np.float32)
        backpointers = np.empty((len(sentence), self.__tag_vocab_size), dtype=np.int32)

        w = hash(sentence[0])  % self.__word_vocab_size
        for j, t in enumerate(self.tags):
            trellis[0, j] = self.__t1_given_t0_table[(t, START)] * \
                self.__w_given_t_table[(w, t)]
            backpointers[0, j] = -1

        for i in range(1, len(sentence)):
            w = hash(sentence[i])  % self.__word_vocab_size
            for j, t in enumerate(self.tags):
                options = []
                for jprev, tprev in enumerate(self.tags):
                    options.append(trellis[i-1, jprev] * \
                        self.__t1_given_t0_table[(t, tprev)] * \
                        self.__w_given_t_table[(w, t)])
                trellis[i, j] = np.max(options)
                backpointers[i, j] = np.argmax(options)

        tagid = np.argmax(trellis[-1])
        tags.append(self.tags[tagid])

        # Recover Tags
        for i in reversed(range(0, len(sentence) - 1)):
            tagid = backpointers[i + 1, tagid]
            tags.append(self.tags[tagid])

        return list(reversed(tags))


# Main

def example(options):

    splits = options.splits

    dataset = nltk.corpus.treebank.tagged_sents()
    train, dev, test = split(dataset, splits)

    hmm = HMM()

    sentence, tags = zip(*train[0])
    print(sentence)
    print(tags)
    print(list(map(lambda x: hash(x) % len(hmm.tags), tags)))

    print(hmm.log_likelihood(sentence, tags))
    print(hmm.log_likelihood(sentence))
    print(hmm.decode(sentence))

    # print('train: {}'.format(len(train)))
    # print('dev: {}'.format(len(dev)))
    # print('test: {}'.format(len(test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--splits', type=str, default='60,10,30')
    options = parser.parse_args()

    example(options)
