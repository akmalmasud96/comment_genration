from utills import *
from src.dataset import *
import pandas as pd
import numpy as np
import json

if __name__ == "__main__":
     
    words_to_index, index_to_words, word_to_vec_map = read_glove_file("./inputs/glove.6B.100d.txt")

    with open('./inputs/train.txt', 'r') as f:
        train_data = np.array(json.loads(f.read()))
    
    with open('./inputs/test.txt', 'r') as f:
        test_data = np.array(json.loads(f.read()))
    # getting max length comment from comments corpus
    maxLen = len(max(train_data, key=len).split())
    
    train_data_generator = KerasBatchGenerator( train_data, batch_size = 20, vocabulary=len(words_to_index), max_len = maxLen,word_to_index = words_to_index)