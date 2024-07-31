

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from hparams import *
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing  

from sklearn.model_selection import TimeSeriesSplit

one_hot_vectorizer = CountVectorizer(binary=True)

def find_NaN_indeces(dataframe):
  return np.unique(np.where(dataframe.isnull())[0])

def remove_Nan_features(df_raw, idx):
  return df_raw.drop(index=idx).reset_index(drop=True)

def df_append(df1:pd.DataFrame, df2:pd.DataFrame, columns):
  return df1[columns].append(df2[columns]).reset_index(drop=True)

def remove_House_labels(dataframe, idx):
  return dataframe.drop(idx).reset_index(drop=True)

def map_labels(labels):
  return one_hot_vectorizer.fit_transform(labels).toarray()

def build_model(input_shape, layers_list, activation):

    model = Sequential()
    model.add(LSTM(layers_list[0], input_shape=input_shape, return_sequences=True))
    layers_list.pop(0)
    for feat_out in layers_list[:-1]:
      model.add(LSTM(feat_out, return_sequences=True))

    model.add(LSTM(layers_list[-1]))
    model.add(Dense(1, activation=activation))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                                    loss=LOSS_FUNC)

    return model

def gen_data_train(index, in_seq_len, out_seq_len, input_array, output_array, batch_size=16):
    
    l = len(index) // batch_size
    X = []
    Y = []
    
    start = index[-1] - out_seq_len - l*batch_size                              
    while start<2*in_seq_len: start += batch_size                                 
    end = index[-1] - out_seq_len 

    while True:
        j = 1
        for i in range(start, end):

            X.append(input_array[index[i - JUMP*in_seq_len:i:JUMP]])
            Y.append(output_array[index[i:i + out_seq_len]])

            if j% batch_size == 0:

                yield np.array(X).astype(np.float32), np.array(Y).astype(np.float32)
                X = []
                Y = []
            j += 1


def gen_data_test(index_in, index_out, in_seq_len, input_array, output_array):
    
    X = input_array[index_in[::JUMP]]
    Y = output_array[index_out[0]]
    return np.expand_dims(X, axis=0).astype(np.float32), np.expand_dims(Y, axis=0).astype(np.float32)
    
def spe(index, in_seq_len, out_seq_len, batch_size=16):
    
    l = len(index) // batch_size   
    start = index[-1] - out_seq_len - l*batch_size
    i = 0
    while start<in_seq_len: 
        start += batch_size
        i+=1
    return l-i    