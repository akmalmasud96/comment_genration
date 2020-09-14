import numpy as np
import pandas as pd


def read_glove_file(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def file_to_word_ids(filename, words_to_index):

    words = []
    with open(filename, "r") as f:
        words = f.read().split()
    return [words_to_index.get(word, words_to_index["unk"]) for word in words]