from utills import *
from src.dataset import *
import pandas as pd
import numpy as np
import json
import keras
from src.Network import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

if __name__ == "__main__":

    num_epochs = 50
    num_steps = 30
    batch_size = 20
     
    words_to_index, index_to_words, word_to_vec_map = read_glove_file("./inputs/glove.6B.50d.txt")

    #test_data = file_to_word_ids("./inputs/test.txt",words_to_index)
    train_data = file_to_word_ids("./inputs/train.txt",words_to_index)
    test_data = file_to_word_ids("./inputs/val.txt",words_to_index)

    train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary=len(words_to_index),
                                           skip_step=num_steps)
    valid_data_generator = KerasBatchGenerator(test_data, num_steps, batch_size, vocabulary=len(words_to_index),
                                           skip_step=num_steps)

    #create model 
    model = My_Model((num_steps,), word_to_vec_map, words_to_index,  hidden_size = 128)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    checkpointer = ModelCheckpoint(filepath='./models/model-{epoch:02d}.h5', verbose=1,save_weights_only=False)
    
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(test_data)//(batch_size*num_steps), callbacks=[checkpointer])