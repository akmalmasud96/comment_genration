import os
import tensorflow as tf
from keras.layers import Dense, Activation, Embedding, Dropout, Input, TimeDistributed
from keras.layers import LSTM
import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 100-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]     # define dimensionality of your GloVe word vectors (= 100)
    
    
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)


    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def My_Model(input_shape, word_to_vec_map, word_to_index, hidden_size = 500):
    """
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 100-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    model -- a model instance in Keras

    """

    vocabulary = len(word_to_index)

    model = Sequential()
    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    model.add (Input(shape=input_shape, dtype=np.int32))
    
    # Create the embedding layer pretrained with GloVe Vectors 
    # Propagate sentence_indices through your embedding layer
    model.add(pretrained_embedding_layer(word_to_vec_map, word_to_index))
    
     
    # Propagate the embeddings through an LSTM layer
    # The returned output should be a batch of sequences.
    model.add(LSTM(hidden_size, return_sequences=True))

    # Add dropout with a probability of 0.5
    model.add(Dropout(0.5))

    # Propagate X trough another LSTM layer
    # The returned output should be a single hidden state, not a batch of sequences.
    model.add(LSTM(hidden_size, return_sequences=True))

    # Add dropout with a probability of 0.5
    model.add(Dropout(0.5))
    
    # Propagate X through a Dense layer with vocabulary units
    model.add(TimeDistributed(Dense(vocabulary)))
    
    # Add a softmax activation
    model.add(Activation('softmax'))
    
    return model