# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:06:06 2020

@author: jonsnow
"""

import matplotlib.pyplot as plt
#import sys
import mplfinance as mpf
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from PIL import Image
import PIL
from keras import optimizers
import tensorflow.keras.backend
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import model_from_json, load_model
from keras.layers import Dropout
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, save_model, load_model
import os
from sklearn.model_selection import train_test_split
from datetime import date, datetime, timedelta
import json
import datetime
import time


frame_size = 100
threshold_Y_binary_percent = 1.024
lower_limit = 0.01
upper_limit = 0.1

#path_prefix = "C:/frais/datasets/stocks_cache/"
path_prefix = ''
path_tickers_list = "./middle_volatility_potential_top100.csv"
#path_stock_data = "stock_data/filtered_vol/l" + str(lower_limit) + '/u' + str(upper_limit)
path_stock_data = "stock_data/filtered_vol_std/l" + str(lower_limit) + '/u' + str(upper_limit)
path_stock_data = path_prefix + path_stock_data
#market_cap_path = "tickers_with_market_cap.csv"
path_cached_df = "C:/frais/datasets/stocks_cache/cached_df/201008_df_before_split"
models_folder = "measured_stats/models_results/trivial_cnn"
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_10_09__02_42_53"
path_vix_daily = "stock_data/vixcurrent_daily.csv"
vix_daily_min = 8.56
vix_daily_mean = 10.323445
colse_divided_by_open_min = 0.5
close_divided_by_subframe_min = 0.8
close_divided_by_subframe_factor = 2.5

#marketcap_df = pd.read_csv(market_cap_path).set_index('symbol')

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value
    
    
def get_clean_X_Y_percentage_from_df_by_marketcap_volume(df_list, frame_length = 100):

    X_list = []
    Y_list = []
    list_to_test = []
    ticker_list = []
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            list_to_test.append((i,j))
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            current_df = df_list[i][j]
            current_day_frame_length = len(current_df)
            if(current_day_frame_length < frame_length):
                continue
#            current_df = current_df[:frame_length].drop('symbol', axis=1)
            symbol = current_df['symbol'][0]
            current_df = current_df.drop('symbol', axis=1)
            y_value = check_df_for_maximum_gain_percent(current_df, frame_length)
            current_df = current_df[:frame_length]
            if(y_value == float('inf')):
                continue
            X_list.append(current_df)
            Y_list.append(y_value)
            ticker_list.append(symbol)
            
#            print(current_df)
    return X_list, Y_list, ticker_list

def my_is_number(var_to_test):
    try:
        var_to_test / 1
        if(var_to_test > 0 or var_to_test < 1):
            return True
        return False
    except:
        return False

def get_stock_data_from_folderlist(folderlist):
    all_df_list = []
#    i = 0
    for path_ticker in folderlist:
        current_df = []
        current_path = path_ticker[1]
#        if (path_ticker[0] == 'BCRX'):
#            print("BCRX at index {}".format(i))
        filelist = [name for name in os.listdir(current_path)]
        if(len(filelist) == 0):
            continue
        try:
            for file_df in filelist:
                current_df.append(pd.read_csv(current_path + file_df).set_index('date'))
            all_df_list.append(current_df)
#            i += 1
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
    return all_df_list

def get_stock_data_folderlist_from_folder(path_all):
    df_files = []
    for path_ticker in get_immediate_subdirectories(path_all):
#        print(path_all+"/"+path_ticker)
        current_path = path_all+"/"+path_ticker+"/"
        filelist = [name for name in os.listdir(current_path)]
#        mydata = ''
        if(len(filelist) == 0):
            continue
        try:
            df_files.append((path_ticker,current_path))
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
    return df_files
    
def get_vix_fdaily_from_file_and_scale(path, min_value, mean):
    vix_daily = pd.read_csv(path, index_col='Date')
    vix_daily = vix_daily - min_value
    vix_daily = vix_daily / (2 * mean)
    return vix_daily

def convert_date_from_df_format_to_vix_format(date_to_convert):
    date_original = datetime.datetime.strptime(date_to_convert.split(' ')[0], "%Y-%m-%d")
    new_year = str(date_original.year)
    new_month = str(int(date_original.month))
    new_day = str(int(date_original.day))
    return new_month + '/' + new_day + '/' + new_year

def get_vix_in_df_by_index(index_in_df, df_list, vix_df):
    date_in_vix = convert_date_from_df_format_to_vix_format(df_list[index_in_df].index[0])
    return vix_df.loc[date_in_vix][0]    


def get_close_divided_by_max_in_single_df(df):
    min_value = df['Low'].min()
    max_value = df['High'].max() - min_value
    close_value = df['Close'][-1:][0] - min_value
    return close_value / max_value

def get_close_divided_by_max_in_df_list(df_list):
    output_list = []
    for df in df_list:
        output_list.append(get_close_divided_by_max_in_single_df(df))
    return output_list

def get_close_divided_by_open_in_single_df(df):
#    min_value = df['Low'].min()
#    open_value = df['Open'].max() - min_value
#    close_value = df['Close'][-1:][0] - min_value
    open_value = df['Open'].max()
    close_value = df['Close'][-1:][0]
    return close_value / open_value

def get_close_divided_by_open_in_df_list(df_list, colse_divided_by_open_min):
    output_list = []
#    i = 0
    for df in df_list:
#        print(i)
        output_list.append(get_close_divided_by_open_in_single_df(df) - colse_divided_by_open_min)
#        i+=1
    return output_list

def get_close_divided_by_open_subframe_in_single_df(df, subframe_length = 30):
#    min_value = df['Low'].min()
#    open_subframe_value = df['Close'][-31:-subframe_length][0] - min_value
#    close_value = df['Close'][-1:][0] - min_value
    open_subframe_value = df['Close'][-31:-subframe_length][0]
    close_value = df['Close'][-1:][0]
    return close_value / open_subframe_value


def get_close_divided_by_open_subframe_in_df_list(df_list, close_divided_by_subframe_min, close_divided_by_subframe_factor, subframe_length = 30):
    output_list = []
    for df in df_list:
        output_list.append(close_divided_by_subframe_factor *
                           (get_close_divided_by_open_subframe_in_single_df(df, subframe_length) - close_divided_by_subframe_min))
    return output_list


#folderlist_stock_data = [('BCRX', 'stock_data/filtered_vol/l1000000/u6000000/BCRX/')]

def get_X_Y_data_from_folderlist(folder_path, frame_length):
    clean_df = get_stock_data_from_folderlist(folder_path)
#    print('getting volume groups df')
    df_X, df_Y, tickers = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, frame_length = frame_length)
#    print('fomatting df')
    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
    return df_X, df_Y_binary, tickers


def get_X_Y_data_from_cached_csv(folder_path, size=0):
    dflist = []
    filelist = [name for name in os.listdir(folder_path+ "/")]
    count_read_files = 0
    if(len(filelist) == 0):
        pass
        #continue
    i = 0
    for file_df in filelist:
        try:
                #current_df.append(pd.read_csv(folder_path + file_df).set_index('date'))
#                print(folder_path + "/" + file_df)
                dflist.append(pd.read_csv(folder_path + "/df_" + str(i) + ".csv", index_col='date'))
                count_read_files += 1
                if(size != 0 and count_read_files >= size):
                    break
                if (count_read_files % 5000 == 0):
                    print("read {} x dataframes".format(count_read_files))
                i += 1
        except KeyboardInterrupt:
            break
            pass
        except:
            print("An exception occurred")
    list_y_values = list(pd.read_csv(folder_path + ".csv")['0'])
    if(size > 0):
        list_y_values = list_y_values[:size]
    return dflist, list_y_values

file_batch_size = 40
df_rawframe_X = []
df_all_Y_binary = []
test_potential_volatility_list = []
tickers_list = []
folderlist_stock_data = get_stock_data_folderlist_from_folder(path_stock_data)[:1900]

for i in range(int(len(folderlist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(folderlist_stock_data) / file_batch_size) + 1))
    df_X, df_Y, tickers = get_X_Y_data_from_folderlist(
            folderlist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(folderlist_stock_data))],
            frame_size)
    for x in df_X : 
        df_rawframe_X.append(x) 
    for y in df_Y : 
        df_all_Y_binary.append(y) 
    for t in tickers : 
        tickers_list.append(t) 

date_list = []
for df in df_rawframe_X:
    date_list.append(df.index[0])
        
df_cached_trivial_frame_X, df_all_Y_binary_temp = get_X_Y_data_from_cached_csv(path_cached_df, size=len(df_rawframe_X))
del(df_all_Y_binary_temp)
model_trivial_dnn = keras.models.load_model(
    path_trivial_cnn +"/model.h5",
    custom_objects=None,
    compile=False
)
#df_rawframe_X
trivial_nn_score = model_trivial_dnn.predict(np.array(df_cached_trivial_frame_X).reshape(len(df_cached_trivial_frame_X), frame_size, 10, 1))
trivial_nn_score =  trivial_nn_score.reshape((len(trivial_nn_score)))
std_of_same_day_list = []
for df in df_rawframe_X:
    std_of_same_day_list.append(df.drop('Volume', axis =1).std().mean() * 3)
vix_daily = get_vix_fdaily_from_file_and_scale(path_vix_daily, vix_daily_min, vix_daily_mean)
vix_in_df_list = []
for i in range(len(df_rawframe_X)):
    vix_in_df_list.append(get_vix_in_df_by_index(i, df_rawframe_X, vix_daily))

print(1)
close_divided_by_max_list = get_close_divided_by_max_in_df_list(df_rawframe_X)
print(2) 
close_divided_by_open_list = get_close_divided_by_open_in_df_list(df_rawframe_X, colse_divided_by_open_min) 
print(3)
close_divided_by_subframe_open_list = get_close_divided_by_open_subframe_in_df_list(df_rawframe_X,
                                                                                    close_divided_by_subframe_min, close_divided_by_subframe_factor,  30)    
print(4)


df_all_X = pd.DataFrame([date_list, trivial_nn_score, std_of_same_day_list, vix_in_df_list,
                         close_divided_by_max_list, close_divided_by_open_list, close_divided_by_subframe_open_list]).T
df_all_X.columns = ['date', 'first_nn_score', 'volatility', 'vix', 'close_percent_of_max',
                    'close_percent_of_open', 'close_percent_of_open_subframe']
df_all_X = df_all_X.set_index('date')

parameter_length = len(df_all_X.columns)
print(5)
#
###########################
########### SCALE
###########################
#
#
#print("scaling")
##for df_list in df_all_X:
#for i in range(len(df_all_X)):
#    high = df_all_X[i].drop(['Volume'], axis=1)['High'].max()
#    low = df_all_X[i].drop(['Volume'], axis=1)['Low'].min()
#    delta = high - low
#    df_all_X[i]['Open'] =  (df_all_X[i]['Open'] - low) / delta
#    df_all_X[i]['High'] =  (df_all_X[i]['High'] - low) / delta
#    df_all_X[i]['Low'] =  (df_all_X[i]['Low'] - low) / delta
#    df_all_X[i]['Close'] =  (df_all_X[i]['Close'] - low) / delta
#    
#    high_volume = df_all_X[i]['Volume'].max()
#    low_volume = df_all_X[i]['Volume'].min()
#    delta_volume = high_volume - low_volume
#    df_all_X[i]['Volume'] =  (df_all_X[i]['Volume'] - low_volume) / delta_volume
##
##for i in range(len(x_data)):
##    x_data[i] =x_data[i] - x_data[i].min()
##    x_data[i] = x_data[i] / x_data[i].max()
#
#
###########################
######### RERRANGE
###########################
#print("rearranging: duplicating columns")
#for i in range(len(df_all_X)):
#    df_all_X[i]['High2'] = df_all_X[i]['High'].copy()
#    df_all_X[i]['Close2'] = df_all_X[i]['Close'].copy()
#    df_all_X[i]['Open2'] = df_all_X[i]['Open'].copy()
#    df_all_X[i]['Low2'] = df_all_X[i]['Low'].copy()
#print("rearranging: switching positions")
#for i in range(len(df_all_X)):
#    df_all_X[i] = df_all_X[i].reindex(columns= ['High', 'High2', 'Close', 'Close2', 'Open', 'Open2', 'Low', 'Low2'])
#print("rearranging: adding padding")
#for i in range(len(df_all_X)):
#    df_all_X[i].insert(loc=0, column='padding 0', value=np.zeros(df_all_X[i].shape[0]))
#    df_all_X[i].insert(loc=9, column='padding 9', value=np.zeros(df_all_X[i].shape[0]))
#
#

##########################
##### TRAIN TEST SPLIT
##########################
    
manual_test_train_split = True

print("train test split")
#df_all_X_np = np.array(df_all_X)
#df_all_X_np = df_all_X_np.reshape(df_all_X_np.shape[0], frame_size, 10, 1)
#X_train, X_test, Y_train, Y_test = train_test_split(df_all_X_np, df_all_Y_binary, test_size=0.30)
#del(df_all_X_np)

#X_train, X_test, Y_train, Y_test = train_test_split(df_all_X, df_all_Y_binary, test_size=0.30)
#
#X_train = np.array(X_train).reshape((len(X_train),3,1))
#X_test = np.array(X_test).reshape((len(X_test),3,1))


#print("train test split by date")
X_train = []
Y_train = []
X_test = []
Y_test = []
date_threshold_train_test = datetime.datetime.now() - timedelta(days=45)
for i in range(len(df_all_X)):
    if (datetime.datetime.strptime(df_all_X.index[i], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
        X_train.append(df_all_X.iloc[i])
        Y_train.append(df_all_Y_binary[i])
    else:
        X_test.append(df_all_X.iloc[i])
        Y_test.append(df_all_Y_binary[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.reshape(X_train.shape[0], parameter_length, 1)
X_test = X_test.reshape(X_test.shape[0], parameter_length, 1)

##########################
### DUPLICATE POSITIVES
###########################

print("duplicating positives")
positive_indices_train = []
for i in range(len(Y_train)):
    if(Y_train[i]):
        positive_indices_train.append(i)

time_to_duplicate = 0
for x in range(time_to_duplicate):
    new_df_x_part = []
    new_df_y_part = []
    for i in positive_indices_train:
        new_df_x_part.append(X_train[i].copy())
        new_df_y_part.append(True)
    new_df_x_part = np.array(new_df_x_part)
    X_train = np.concatenate([X_train, new_df_x_part])
    for y in new_df_y_part : 
        Y_train.append(True) 
##########################
########## TRAIN
##########################

import keras.backend as K
def precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  
    precision = true_positives / (predicted_positives + K.epsilon())
#    precision = true_positives / (predicted_positives)
    return precision
def recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
#    recall = true_positives / (possible_positives)
    return recall

model_binary = tf.keras.models.Sequential([
# YOUR CODE HERE
        tf.keras.layers.Dense(10, activation='relu', input_shape = (parameter_length,1)),
#        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dropout(0.1),
#        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='relu'),
#        tf.keras.layers.Dropout(0.15),
        # tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
#        tf.keras.layers.Dense(2, activation='softmax')
#       tf.keras.layers.Dense(1, tf.keras.activations.tanh)
])

#model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy',  metrics=['accuracy', precision, recall])

#model_binary.compile( optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8),
#                     loss='categorical_crossentropy',  metrics=['accuracy', precision, recall])
#model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                     loss='mse',  metrics=['accuracy', precision, recall])

learning_rate = 0.00003
epochs = 15
model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='mse',  metrics=['accuracy', precision, recall])
#history_binary = model_binary.fit(X_train,np.array(Y_train),verbose=1,batch_size=64, shuffle=True,
#                                  epochs=epochs,validation_data=(X_test,np.array(Y_test)))
#history_binary = model_binary.fit(np.array(X_train),np.array(Y_train),verbose=1,batch_size=64, shuffle=True,
#                                  epochs=epochs,validation_data=(np.array(Y_test),np.array(Y_test)))
history_binary = model_binary.fit(np.asarray(X_train, dtype=float), np.array(Y_train),verbose=1,batch_size=20, shuffle=True,
                                  epochs=epochs,validation_data=(np.asarray(X_test, dtype=float),np.array(Y_test)))

predictions = model_binary.predict(np.asarray(X_test, dtype=float))

threshold = 0.5
p = predictions > threshold


c = 0
for i in range(len(p)):
    if(p[i]):
        if(Y_test[i] == True):
            c += 1
#
#print("with threshold {} : {} out of {} are correct".format(threshold, c, p.sum()))

def result_forThreshold(predictions, threshold):
    p = predictions > threshold
    
    
    c = 0
    for i in range(len(p)):
        if(p[i]):
            if(Y_test[i] == True):
                c += 1
    return c , p.sum()
#print("with threshold {} : {}".format(threshold, result_forThreshold(predictions, threshold)))
#print("with threshold {} : {}".format(0.6, result_forThreshold(predictions, 0.6)))
#print("with threshold {} : {}".format(0.7, result_forThreshold(predictions, 0.7)))
#print("with threshold {} : {}".format(0.8, result_forThreshold(predictions, 0.8)))
#print("with threshold {} : {}".format(0.9, result_forThreshold(predictions, 0.9)))
    
thresholds = np.linspace(0.5, 0.9, 5)
thresholds_results = []

for i in range(len(thresholds)):
    threshold = thresholds[i]
    result = result_forThreshold(predictions, threshold)
    thresholds_results.append(result)

result_05 = result_forThreshold(predictions, 0.5)

for i in range(len(thresholds)):
    correct_guessus = thresholds_results[i][0]
    total_gueses = thresholds_results[i][1]
    print("with threshold {} : {} out of {} are correct. percantage: {:.2f}%".format(thresholds[i],
          correct_guessus, total_gueses, 100 * correct_guessus / total_gueses))
#
#model = model_binary2.summary()
#frame_size = frame_size
#train_len = len(X_train)
#test_len = len(X_test)
#taining_epochs = epochs
#learning_rate = learning_rate
#upper_limit = upper_limit
#lower_limit = lower_limit
#description = ''
#loss_function = 'mse'
#optimizer = 'Adam'
#history =  history_binary2.history
#thresholds = thresholds
#thresholds_results = thresholds_results
#last_val_precision = history_binary2.history['val_precision'][-1:][0]
#last_val_recall = history_binary2.history['val_recall'][-1:][0]
#last_percentage = 100 * result_05[0] / result_05[1]
#
#results_dict = {
#        'model' : str(model),
#        'frame_size' : str(frame_size),
#        'train_len' : str(train_len),
#        'test_len' : str(test_len),
#        'taining_epochs' : str(taining_epochs),
#        'learning_rate' : str(learning_rate),
#        'upper_limit' : str(upper_limit),
#        'lower_limit' : str(lower_limit),
#        'description' : str(description),
#        'loss_function' : str(loss_function),
#        'optimizer' : str(optimizer),
#        'history' :  str(history),
#        'thresholds' : str(thresholds),
#        'thresholds_results' : str(thresholds_results),
#        'last_val_precision' : str(last_val_precision),
#        'last_val_recall' : str(last_val_recall)
#        }
#
#current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
#output_save_model_folder = models_folder + "/" + current_time + "/"
#if not os.path.exists(output_save_model_folder):
#    os.mkdir(output_save_model_folder)
#
#model_binary2.save(output_save_model_folder + 'model.h5')
#model_binary2.save_weights(output_save_model_folder + 'weights.h5')
#
#import json
#json_results = json.dumps(results_dict)
#f = open(output_save_model_folder + 'results.json',"w")
#f.write(json_results)
#f.close()
