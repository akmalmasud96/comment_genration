from utills import *
from src.dataset import *
import pandas as pd
import numpy as np
import json
from src.Network import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":

    num_epochs = 10
    batch_size = 20
     
    words_to_index, index_to_words, word_to_vec_map = read_glove_file("./inputs/glove.6B.100d.txt")

    with open('./inputs/train.txt', 'r') as f:
        train_data = json.loads(f.read())
    
    with open('./inputs/test.txt', 'r') as f:
        test_data = json.loads(f.read())
    # getting max length comment from comments corpus
    maxLen = 100
    
    train_data_indices, y_train_indices= sentences_to_indices ( np.array(train_data[:10]), word_to_index = words_to_index, max_len =maxLen, vocabulary=len(words_to_index))
    test_data_indices, y_test_indices= sentences_to_indices ( np.array(test_data[:10]), word_to_index = words_to_index, max_len =maxLen, vocabulary=len(words_to_index))

    #create model 
    model = My_Model((maxLen,), word_to_vec_map, words_to_index,  hidden_size = 128)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    checkpointer = ModelCheckpoint(filepath='./model-{epoch:02d}.hdf5', verbose=1)
    
    model.fit(train_data_indices, y_train_indices, epochs = 5, batch_size = 10,validation_data=(test_data_indices, y_test_indices), shuffle=True, callbacks=[checkpointer])