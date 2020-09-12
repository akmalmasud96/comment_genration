from utills import *
import pandas as pd
import numpy as np

if __name__ == "__main__":
     
    # genrate Comments.csv 
    # read_all_csv("inputs")

    #words_to_index, index_to_words, word_to_vec_map = read_glove_file("./inputs/glove.6B.100d.txt")

    comments = pd.read_csv("./inputs/Comment.csv")
    comments = pre_processing(comments)
    
    # getting max length comment from comments corpus
    maxLen = len(max(comments, key=len).split())
    

    