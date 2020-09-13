from keras.utils import to_categorical
import numpy as np


def sentences_to_indices(X, word_to_index, max_len, vocabulary):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    y_indices -- 
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros([m,max_len])
    y_indices = np.zeros((m, max_len, vocabulary))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index.get(w, word_to_index["unk"])
            # Increment j to j + 1
            j = j+1

            if j >= max_len:
                break

        temp_y = X_indices[i,1:]
        # making array equal to max size
        temp_y = np.append(temp_y,0)
        # convert all of temp_y into a one hot representation
        y_indices[i, :, :] = to_categorical(temp_y, num_classes=vocabulary)
    
    return X_indices, y_indices


class KerasBatchGenerator(object):
    def __init__(self, data, batch_size, vocabulary,word_to_index, max_len ):
        self.data = data
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.word_to_index = word_to_index
        self.max_len = max_len

        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0


    def generate(self):
        x = np.zeros((self.batch_size, self.max_len))
        y = np.zeros((self.batch_size, self.max_len, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                
                # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
                sentence_words = self.data[self.current_idx].lower().split()

                # Initialize j to 0
                j = 0

                # Loop over the words of sentence_words
                for w in sentence_words:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    x[i, j] = self.word_to_index.get(w, self.word_to_index["unk"])
                    # Increment j to j + 1
                    j = j+1

                    if j >= self.max_len:
                        break
            
                temp_y = x[i,1:]
                # making array equal to max size
                temp_y = np.append(temp_y,0)
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += 1
            yield x, y
    