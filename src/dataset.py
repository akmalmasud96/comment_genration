from keras.utils import to_categorical
import numpy as np


class KerasBatchGenerator(object):
    def __init__(self, data, batch_size, vocabulary, max_len, word_to_index):
        self.data = data
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.word_to_index = word_to_index
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0

    def generate(self):
        
        x = np.zeros((self.batch_size, self.max_len))
        y = np.zeros((self.batch_size, self.max_len, self.vocabulary))

        while True:

            for i in range(self.batch_size):

                sentence_words = self.data[i].lower().split()
                # Initialize j to 0
                j = 0
                # Loop over the words of sentence_words
                for w in sentence_words:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    x[i, j] = self.word_to_index[w]
                    
                    # Increment j to j + 1
                    j = j+1
                    
                    # break loop when j is equal to max length
                    if j == self.max_len:
                        break
                    

                temp_y = x[i,1:]
                # making array equal to max size
                temp_y = np.append(temp_y,0)
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                
            yield x, y