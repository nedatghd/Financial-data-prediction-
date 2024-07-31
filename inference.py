
from utils import *
import pandas as pd
from hparams import *
from keras.models import load_model
import train
import fetch_data
import argparse

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data', default=PATH_TO_DATA, type=str) 
parser.add_argument('--epoch_number', default=152262, type=int)

saved_data = pd.read_excel(PATH_TO_DATA)
idx = find_NaN_indeces(saved_data)
saved_df_removed_NaN = remove_Nan_features(saved_data, idx)

current_epoch = fetch_data.get_current_epoch()
query_data = fetch_data.get_data(last_epoch_num=saved_data['epoch'].iloc[-1], current_epochNum=current_epoch).iloc[:-2]
idx = find_NaN_indeces(query_data)
query_df_removed_NaN = remove_Nan_features(query_data, idx)
query_df_removed_NaN = query_df_removed_NaN.rename(columns={'position':'Up_Down'})

saved_and_query_df = df_append(saved_df_removed_NaN, query_df_removed_NaN, columns=['epoch', 'Up_Down', 'closePrice'])

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

idx = np.where(saved_and_query_df == 'House')[0]
df_removed_House = remove_House_labels(saved_and_query_df, idx)

df_features_input = pd.Series(df_removed_House['closePrice'])
df_label_output = pd.Series(df_removed_House['Up_Down'])

df_input_scaled = min_max_scaler.fit_transform(np.expand_dims(df_features_input, axis=1))
df_label_encoded_output = map_labels(df_label_output)

print('start training')
# model = train.run(df_input_scaled, df_input_scaled, model=True, verbose=1, epoch=2)
model = load_model(PATH_TO_LAST_CHKPOINT)

print('start inference')
tscv = TimeSeriesSplit(n_splits=N_SPLIT_INFERENCE, test_size=TEST_SIZE_IN_TSCV)

for train_index, test_index in tscv.split(df_features_input):

    input_features = np.expand_dims(df_input_scaled[train_index[-JUMP*INPUT_SEQUENCE_LENGTH::JUMP]], axis=0)
    pred_price = model.predict(input_features, verbose=0).squeeze()
    pred_label = (pred_price >= df_input_scaled[train_index[-1]])[0]

    pred_label_to_str = ['Bull' if pred_label else 'Bear']
    true_label = df_label_output.iloc[test_index[-1]]

    last_idx = df_label_output.index[-1]
    with open('output.txt', 'w') as f:
        f.write('round number: {}\n'.format(df_removed_House['epoch'].iloc[test_index[-1]]))
        # print("predicted label:", min_max_scaler.inverse_transform(df_input_scaled[train_index[-1]].reshape(1,-1)))
        f.write('predicted label: {}'.format(pred_label_to_str))