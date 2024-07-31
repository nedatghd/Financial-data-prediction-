

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from utils import *
from keras.models import load_model

# import hyperparamters and path of files 
from hparams import *

def running_data():
    # data
    df = pd.read_excel(PATH_TO_DATA)
    # df = df.iloc[15000:].reset_index()

    idx = find_NaN_indeces(df)
    df_removed_NaN = remove_Nan_features(df, idx)

    idx = np.where(df_removed_NaN == 'House')[0]
    df_removed_House = remove_House_labels(df_removed_NaN, idx)

    df_features_input = df_removed_House['closePrice']

    min_max_scaler = preprocessing.MinMaxScaler((0,1))
    # df_features_input_diff = pd.Series(df_removed_House['closePrice']-df_removed_House['lockPrice'])

    df_input_scaled = min_max_scaler.fit_transform(np.expand_dims(df_features_input, axis=1))
    # df_label_output = pd.Series(df_removed_House['Up_Down'])
    # df_label_encoded_output = map_labels(df_label_output)[:,0]

    df_output_scaled = df_input_scaled.copy()
    return df_input_scaled, df_output_scaled

def run(data_in, data_out, model:bool=False, verbose:int=0, epoch:int=5):
    """
    run training on data (from hparams.PATH_TO_DATA)
    the training has callback to save best model
    return trained model
    """

    # model    
    if model:
        model = load_model(PATH_TO_LAST_CHKPOINT)
        callbacks_test = []
        n_split_train = 2 
    else: 
        model = build_model(input_shape=INPUT_SHAPE, layers_list=LAYERS_LIST, activation=ACTIVATION)
        callbacks_test = [keras.callbacks.ModelCheckpoint(
                PATH_TO_BEST_CHKPOINT, save_best_only=True, monitor="loss"),
                keras.callbacks.EarlyStopping(monitor="loss", patience=200, verbose=1),
                keras.callbacks.TensorBoard(log_dir=PATH_TO_LOG)]
        n_split_train = N_SPLIT_TRAIN

    # train
    tscv = TimeSeriesSplit(n_splits=n_split_train, test_size=TEST_SIZE_IN_TSCV)

    for train_index, test_index in tscv.split(data_in):

        data_train_loader = gen_data_train(index=train_index,
                                        in_seq_len=INPUT_SEQUENCE_LENGTH,
                                        out_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                        input_array=data_in,
                                        output_array=data_out,
                                        batch_size=BATCH_SIZE)

        data_test = gen_data_test(index_in=train_index[-JUMP*INPUT_SEQUENCE_LENGTH:],
                                index_out=test_index,
                                in_seq_len=INPUT_SEQUENCE_LENGTH,
                                input_array=data_in,
                                output_array=data_out)
        
        step_per_epoch = spe(train_index, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, batch_size=BATCH_SIZE)     

        model.fit(
            data_train_loader, 
            epochs = epoch, 
            steps_per_epoch = step_per_epoch,
            validation_data = data_test,
            verbose = verbose, 
            callbacks = callbacks_test, 
            validation_steps = 1)
    return model


if __name__ == "__main__":
    data_input, data_output = running_data()
    model = run(data_input, data_output, verbose=1, epoch=EPOCH)
    model.save(PATH_TO_LAST_CHKPOINT)