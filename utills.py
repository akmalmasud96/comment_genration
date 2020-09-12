import numpy as np
import pandas as pd
import glob
import string


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



def read_all_csv(files_path):
    all_files = glob.glob(files_path + "/*.csv")

    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0,usecols=["commentID","commentBody","sectionName"])
        df_list.append(df)

    data_frame = pd.concat(df_list, axis=0, ignore_index=True)
    data_frame.drop_duplicates(subset='commentID', inplace=True)
    
    #only selecting Politics comments and returning commentBody only
    data_frame =  data_frame[data_frame["sectionName"]=="Politics"]["commentBody"]

    data_frame.to_csv(files_path+"/Comment.csv")

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 


def pre_processing(comments):

    comments = comments["commentBody"].values.tolist()
    comments = [clean_text(x) for x in comments]

    return comments