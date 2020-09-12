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
        train_data = np.array(json.loads(f.read()))
    
    with open('./inputs/test.txt', 'r') as f:
        test_data = np.array(json.loads(f.read()))
    # getting max length comment from comments corpus
    maxLen = len(max(train_data, key=len).split())
    
    train_data_generator = KerasBatchGenerator( train_data[30], batch_size = 20, vocabulary=len(words_to_index), max_len = maxLen,word_to_index = words_to_index)
    valid_data_generator = KerasBatchGenerator( test_data[30], batch_size = 20, vocabulary=len(words_to_index), max_len = maxLen,word_to_index = words_to_index)

    #create model 
    model = My_Model((maxLen,), word_to_vec_map, words_to_index,  hidden_size = 128)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    checkpointer = ModelCheckpoint(filepath='./model-{epoch:02d}.hdf5', verbose=1)

    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(test_data)//(batch_size), callbacks=[checkpointer])