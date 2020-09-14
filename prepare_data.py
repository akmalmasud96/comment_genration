import numpy as np
import pandas as pd
import glob
import json
import string
from sklearn.model_selection import train_test_split

def read_all_csv(files_path):
    all_files = glob.glob(files_path + "*.csv")

    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0,usecols=["commentID","commentBody","sectionName"])
        df_list.append(df)

    data_frame = pd.concat(df_list, axis=0, ignore_index=True)
    data_frame.drop_duplicates(subset='commentID', inplace=True)
    
    #only selecting Politics comments and returning commentBody only
    data_frame =  data_frame[data_frame["sectionName"]=="Politics"]["commentBody"]
    return data_frame


def clean_text(txt,pun):
    txt = txt.replace('US','america')
    txt = txt.replace('<br/>','').lower()
    txt = txt.replace('u.s','america')
    txt = txt.replace('mr.','mr')
    txt = txt.replace('mrs.','mr')
    txt = txt.replace('...','.')
    txt = "".join(v for v in txt if v not in pun)

    txt = txt.translate(str.maketrans({key: " {0} ".format(key) for key in ',!.?'})).strip()
    return txt


def pre_processing(data,pun):

    comments = data.values.tolist()
    comments = [clean_text(x,pun) for x in comments]
    return comments


def split_data(data):

    train_data, test_data = train_test_split(data,test_size=40, random_state=42)
    test_data, val_data = train_test_split(test_data,test_size=10, random_state=42)

    with open('./inputs/train.txt', 'w') as f:
        f.write(" ".join(train_data))
    
    with open('./inputs/test.txt', 'w') as f:
        f.write(" ".join(test_data))
    
    with open('./inputs/val.txt', 'w') as f:
        f.write(" ".join(val_data))


if __name__ == '__main__':

    pun = '\"#$%&\'()*+=-/:;<=>@[\]^_`{|}~'

    comments = read_all_csv("inputs/")
    print(comments.head())

    process_comments = pre_processing(comments,pun)

    split_data(process_comments)