import pickle
import re
import numpy as np
from ewc_app.const.constants import *

class VocabLoader:
    oov_std = 0.001
    VOCAB_SIZE = 50000
    _vocab = None
    _word_count = None

    @staticmethod
    def load():
        if not VocabLoader._vocab:

            VocabLoader._vocab = Vocab(VOCAB_PATH, VocabLoader.VOCAB_SIZE)
        return VocabLoader._vocab


    @staticmethod
    def get_embedding_matrix():
        path = EMBEDDING_PATH
        # np.load is only a pointer on a file
        mmap_embedding = np.load(path, mmap_mode="r")
        # we have to clone the embedding in order to use it later. .copy is mandatory
        embedding = mmap_embedding[:VocabLoader.VOCAB_SIZE + 1, :].copy()
        # close the connection to mmap_embedding
        del mmap_embedding
        # add a row for unknown words
        embedding[-1, :] = np.random.normal(0, VocabLoader.oov_std, (1, 300))
        return embedding

    @staticmethod
    def clean_sentence(text, remove_question_mark=False):
        """
        Text preprocessing
        """
        filters = list("!\"#€$%&()*+,-./:;<=>@[\\]^_`{|}~") + ["\t", "\n"]
        temp = text.lower()
        for sign in filters:
            temp = temp.replace(sign, " ")
        # Remove multiple ??
        temp = re.sub(r"\?+", "?", temp)
        if remove_question_mark:
            temp = temp.replace("?", "")
        # Remove multiple spaces
        temp = re.sub(' +', ' ', temp)
        return temp.strip()


class Vocab:
    """
    This class aims at loading the french vocabulary, and implements a couple of convenient functions to deal with
    the vocabulary for embeddings
    """

    def __init__(self, savepath, vocab_size=None, word_shape=None):
        self.savepath = savepath
        with open(self.savepath, "rb") as vocab_file:
            self.vocab = pickle.load(vocab_file)
        self.size = vocab_size if vocab_size else len(self.vocab)
        self.word_shape = word_shape if word_shape else 300
        self.counts_to_index = {}
        self.index_to_counts = {}
        self.index_to_word = {}
        for i in self.vocab.keys():
            self.index_to_counts[self.vocab.get(i).index] = self.vocab.get(i).count
            self.counts_to_index[self.vocab.get(i).count] = self.vocab.get(i).index
            self.index_to_word[self.vocab.get(i).index] = i

    def __contains__(self, item):
        return item in self.vocab

    def get(self, word):
        return self.vocab.get(word)

    def get_word_count(self, word):
        return self.vocab.get(word).count

    def get_word_index(self, word):
        return self.vocab.get(word).index

    def get_index_count(self, index):
        return self.index_to_counts[index]

    def get_index_word(self, index):
        return self.index_to_word[index]

    def print_index_sequence(self, sequence):
        return " ".join([self.get_index_word(i) for i in sequence])

    def get_random_word(self):
        return np.random.normal(0, VocabLoader.oov_std, self.word_shape)

    def encode_sentence(self, sentence):
        sequence = []
        for word in VocabLoader.clean_sentence(sentence).split():
            item = self.vocab.get(word)
            if item:
                # Out of vocabulary words are all assigned to the last index
                sequence.append(min(item.index, self.size))
            else:
                sequence.append(self.size)
        return sequence
