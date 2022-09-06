# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:06:06 2020

@author: jonsnow
"""

import matplotlib.pyplot as plt
#import sys
#import mplfinance as mpf
import tensorflow as tf
import numpy as np
import pandas as pd
#from pandas.plotting import scatter_matrix
#from PIL import Image
#import PIL
from keras import optimizers
import tensorflow.keras.backend
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import model_from_json, load_model
from keras.layers import Dropout
from keras.utils import to_categorical
from numpy import genfromtxt

import os
from sklearn.model_selection import train_test_split
from datetime import date, datetime, timedelta
import json
#import datetime
import time


frame_size = 100
threshold_Y_binary_percent = 1.024
lower_limit = 0.9
upper_limit = 2.9

#path_prefix = "C:/frais/datasets/stocks_cache/"
path_prefix = ''
#path_tickers_list = "./middle_volatility_potential_top100.csv"
#path_stock_data = "stock_data/filtered_vol/l" + str(lower_limit) + '/u' + str(upper_limit)
path_stock_data = "stock_data/filtered_vol_std/l" + str(lower_limit) + '/u' + str(upper_limit)
#path_stock_data ="/home/napol/cached_stocks/std_filtered/l" + str(lower_limit) + '/u' + str(upper_limit)
path_stock_data = path_prefix + path_stock_data
#market_cap_path = "tickers_with_market_cap.csv"
#path_cached_df = "/home/napol/cached_stocks/cached_df/201007_df_before_split"
models_folder = "measured_stats/models_results/trivial_cnn"
tf.config.list_physical_devices(device_type='GPU')

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
            current_df = current_df.drop('symbol', axis=1)
            y_value = check_df_for_maximum_gain_percent(current_df, frame_length)
            current_df = current_df[:frame_length]
            if(y_value == float('inf')):
                continue
            X_list.append(current_df)
            Y_list.append(y_value)
            
#            print(current_df)
    return X_list, Y_list        

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
                current_df.append(pd.read_csv(current_path + file_df).set_index('Date'))
            all_df_list.append(current_df)
#            i += 1
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
    return all_df_list


def get_stock_data_from_folderlist_new(folderlist):
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
#                current_df.append(pd.read_csv(current_path + file_df).set_index('Date'))
#                my_data = genfromtxt((current_path + file_df), delimiter=',')[1:].T[1:-1].T
                current_df.append(genfromtxt((current_path + file_df), delimiter=',')[1:].T[1:-1].T)
#                my_data = my_data[1:].T[1:-1].T
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

folderlist_stock_data = get_stock_data_folderlist_from_folder(path_stock_data)
folderlist_stock_data = folderlist_stock_data[:220]
#folderlist_stock_data = folderlist_stock_data[:100]
#folderlist_stock_data = [('BCRX', 'stock_data/filtered_vol/l1000000/u6000000/BCRX/')]

def get_X_Y_data_from_folderlist(folder_path, frame_length):
    clean_df = get_stock_data_from_folderlist(folder_path)
    print('getting volume groups df')
    df_X, df_Y = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, frame_length = frame_length)
    print('fomatting df')
    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
    return df_X, df_Y, df_Y_binary


def get_X_Y_data_from_cached_csv(folder_path):
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
                if (count_read_files % 5000 == 0):
                    print("read {} x dataframes".format(count_read_files))
                i += 1
        except KeyboardInterrupt:
            break
            pass
        except:
            print("An exception occurred")
    list_y_values = list(pd.read_csv(folder_path + ".csv")['0'])
    return dflist, list_y_values

file_batch_size = 40
df_all_X = []
df_all_Y = []
df_all_Y_binary = []
test_potential_volatility_list = []

start_process_time = datetime.now()
for i in range(int(len(folderlist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(folderlist_stock_data) / file_batch_size) + 1))
    df_X, df_Y, df_Y_binary = get_X_Y_data_from_folderlist(
            folderlist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(folderlist_stock_data))],
            frame_size)
    for x in df_X : 
        df_all_X.append(x) 
    for y in df_Y : 
        df_all_Y.append(y) 
    for y in df_Y_binary : 
        df_all_Y_binary.append(y) 
end_process_time = datetime.now()
print(end_process_time - start_process_time)

date_split_train_test = datetime.strptime("2020-09-01", "%Y-%m-%d")
df_dates_list = []
for df in df_all_X:
    df_dates_list.append(datetime.strptime(df.index[0].split(' ')[0], "%Y-%m-%d"))

split_date_threshold_list = []
for df in df_all_X:
    split_date_threshold_list.append(datetime.strptime(df.index[0].split(' ')[0], "%Y-%m-%d") >= date_split_train_test)
        
#df_all_X, df_all_Y_binary = get_X_Y_data_from_cached_csv(path_cached_df)

#for df in df_all_X:
#    df.to_csv("")

##########################
########## SCALE
##########################


#start_process_time = datetime.now()
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
#end_process_time = datetime.now()
#
#end_process_time - start_process_time

start_process_time = datetime.now()
print("adding bollinger bands")
period = 20
multiplier = 2
for i in range(len(df_all_X)):
    df = df_all_X[i]
    df['UpperBand_Width2'] = df['Close'].rolling(period).mean() + df['Close'].rolling(period).std() * 2
    df['LowerBand_Width2'] = df['Close'].rolling(period).mean() - df['Close'].rolling(period).std() * 2
    df['UpperBand_Width3'] = df['Close'].rolling(period).mean() + df['Close'].rolling(period).std() * 3
    df['LowerBand_Width3'] = df['Close'].rolling(period).mean() - df['Close'].rolling(period).std() * 3
    df_all_X[i] = df.fillna(0)
end_process_time = datetime.now()

print(end_process_time - start_process_time)



start_process_time = datetime.now()
print("scaling and rearranging")
#for df_list in df_all_X:
#start_process_time = datetime.now()
for i in range(len(df_all_X)):
    df_all_X[i] = np.array(df_all_X[i])
    high = df_all_X[i].T[0].max()
    low = df_all_X[i].T[1].min()
    delta = high - low
    high_col = (df_all_X[i].T[0] - low) / delta
    low_col = (df_all_X[i].T[1] - low) / delta
    open_col = (df_all_X[i].T[2] - low) / delta
    close_col = (df_all_X[i].T[3] - low) / delta
    
    bollinger_band_width_2_upper = (df_all_X[i].T[5] - low) / delta
    bollinger_band_width_2_lower = (df_all_X[i].T[6] - low) / delta
    bollinger_band_width_3_upper = (df_all_X[i].T[7] - low) / delta
    bollinger_band_width_3_lower = (df_all_X[i].T[8] - low) / delta
    
    high_volume = df_all_X[i].T[4].max()
    low_volume = df_all_X[i].T[4].min()
    delta_volume = high_volume - low_volume
    volume_col = (df_all_X[i].T[4] - low_volume) / delta_volume
    
#    df_all_X[i] = np.array([high_col, low_col, open_col, close_col, volume_col]).T
    
    padding_col = np.zeros(df_all_X[i].shape[0])
    df_all_X[i] = np.array([padding_col, bollinger_band_width_3_upper, bollinger_band_width_2_upper, high_col, high_col, close_col, close_col, open_col, open_col,
            low_col, low_col, bollinger_band_width_2_lower, bollinger_band_width_3_lower, padding_col, volume_col, volume_col, padding_col]).T
    
df_all_X = np.array(df_all_X)
end_process_time = datetime.now()


print(end_process_time - start_process_time)


for i in range(len(df_all_X)):
    save_cache_path = path_cached_df + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d") + "/df_" + str(i) + ".csv"
    np.savetxt(save_cache_path, df_all_X[i], delimiter=',')
np.savetxt(path_cached_df + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d") + ".csv", df_all_Y, delimiter=',')
np.savetxt(path_cached_df + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d") + "_binary.csv", df_all_Y_binary, delimiter=',')
#
#for i in range(len(x_data)):
#    x_data[i] =x_data[i] - x_data[i].min()
#    x_data[i] = x_data[i] / x_data[i].max()

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
#end_process_time = datetime.now()



##########################
##### TRAIN TEST SPLIT
##########################
    
manual_test_train_split = True
frame_width = df_all_X.shape[2]

#print("train test split")
#df_all_X_np = np.array(df_all_X)
#df_all_X_np = df_all_X_np.reshape(df_all_X_np.shape[0], frame_size, frame_width, 1)
#X_train, X_test, Y_train, Y_test = train_test_split(df_all_X_np, df_all_Y_binary, test_size=0.30)
#del(df_all_X_np)


print("train test split by date")
X_train = []
Y_train = []
X_test = []
Y_test = []
date_threshold_train_test = datetime.now() - timedelta(days=90)
for i in range(len(df_all_X)):
#    if (datetime.strptime(df_all_X[i].index[0], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
    if (split_date_threshold_list[i] ==  False):
        X_train.append(df_all_X[i])
        Y_train.append(df_all_Y_binary[i])
    else:
        X_test.append(df_all_X[i])
        Y_test.append(df_all_Y_binary[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.reshape(X_train.shape[0], frame_size, frame_width, 1)
X_test = X_test.reshape(X_test.shape[0], frame_size, frame_width, 1)

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
    return precision
def recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

model_binary2 = tf.keras.models.Sequential([
# YOUR CODE HERE
        tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape = (frame_size,frame_width,1)),
        # tf.keras.layers.Conv2D(64, (1,3), activation='relu'),
        tf.keras.layers.Conv2D(64, (1,2), activation='relu'),
#        tf.keras.layers.Conv2D(64, (1,5), activation='relu'),
        tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        tf.keras.layers.Conv2D(16, (2,3), activation='relu'),
        tf.keras.layers.Conv2D(32, (1,2), activation='relu'),
        tf.keras.layers.Conv2D(64, (2,3), activation='relu'),
#        tf.keras.layers.Conv2D(32, (1,3), activation='relu'),
        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dropout(0.15),
        #tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
#        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dropout(0.3),
#        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
#        tf.keras.layers.Dropout(0.15),
         tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
#        tf.keras.layers.Dense(2, activation='softmax')
#       tf.keras.layers.Dense(1, tf.keras.activations.tanh)
])

#model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy',  metrics=['accuracy', precision, recall])

#model_binary.compile( optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8),
#                     loss='categorical_crossentropy',  metrics=['accuracy', precision, recall])
#model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                     loss='mse',  metrics=['accuracy', precision, recall])

learning_rate = 0.0001
epochs = 30
model_binary2.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='mse',  metrics=['accuracy', precision, recall])
#model_binary.compile( optimizer="adam", loss=precision,  metrics=['accuracy', precision, recall])
#history_binary = model_binary.fit(np.array(X_train),np.array(Y_train),verbose=1,batch_size=32,  epochs=50,validation_data=(X_test,Y_test))
#history_binary = model_binary.fit(X_train,to_categorical(np.array(Y_train)),verbose=1,batch_size=32, shuffle=True,
#                                  epochs=30,validation_data=(X_test,to_categorical(np.array(Y_test))))
history_binary2 = model_binary2.fit(X_train,np.array(Y_train),verbose=1, steps_per_epoch=2000,batch_size=20, shuffle=True,
                                  epochs=epochs,validation_data=(X_test,np.array(Y_test)))
#history_binary = model_binary2.fit(X_train,np.array(Y_train),verbose=1,batch_size=64, shuffle=True,
#                                  epochs=50,validation_data=(X_test,np.array(Y_test)))

#test_date_list = pd.to_datetime(pd.read_csv("examples/fetched/ZM/2020_04_16.csv").date[:25])

predictions = model_binary2.predict(X_test)

threshold = 0.5
p = predictions > threshold


c = 0
for i in range(len(p)):
    if(p[i]):
        if(Y_test[i] == True):
            c += 1

print("with threshold {} : {} out of {} are correct".format(threshold, c, p.sum()))

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
#thresholds.c
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

model_summary = model_binary2.summary()
frame_size = frame_size
train_len = len(X_train)
test_len = len(X_test)
taining_epochs = epochs
learning_rate = learning_rate
upper_limit = upper_limit
lower_limit = lower_limit
threshold_Y_binary_percent = threshold_Y_binary_percent
description = ''
loss_function = 'mse'
optimizer = 'Adam'
history =  history_binary2.history
thresholds = thresholds
thresholds_results = thresholds_results
last_val_precision = history_binary2.history['val_precision'][-1:][0]
last_val_recall = history_binary2.history['val_recall'][-1:][0]
last_percentage = 100 * result_05[0] / result_05[1]

results_dict = {
        'model' : str(model_summary),
        'frame_size' : str(frame_size),
        'train_len' : str(train_len),
        'test_len' : str(test_len),
        'taining_epochs' : str(taining_epochs),
        'learning_rate' : str(learning_rate),
        'upper_limit' : str(upper_limit),
        'lower_limit' : str(lower_limit),
        'threshold_Y_binary_percent' : str(threshold_Y_binary_percent),
        'description' : str(description),
        'loss_function' : str(loss_function),
        'optimizer' : str(optimizer),
        'history' :  str(history),
        'thresholds' : str(thresholds),
        'thresholds_results' : str(thresholds_results),
        'last_val_precision' : str(last_val_precision),
        'last_val_recall' : str(last_val_recall)
        }

current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
output_save_model_folder = models_folder + "/" + current_time + "/"
if not os.path.exists(output_save_model_folder):
    os.mkdir(output_save_model_folder)

model_binary2.save(output_save_model_folder + 'model.h5')
model_binary2.save_weights(output_save_model_folder + 'weights.h5')

import json
json_results = json.dumps(results_dict)
f = open(output_save_model_folder + 'results.json',"w")
f.write(json_results)
f.close()




#
#import pickle
##dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
#f = open(output_save_model_folder + 'results.txt',"wb")
#pickle.dump(results_dict,f)
#f.close()
#            
#true_indices = []
#minimum_percent = 0.3
#for i in range(len(predictions)):
#    if(predictions[i] > minimum_percent):
#        true_indices.append(i)
#     
#items_to_ckeck = [i for i in true_indices if i < limit_X]
#print(items_to_ckeck)
#for i in items_to_ckeck:
#    # p = pd.DataFrame(x_data[true_indices[0]].T[0].T)
#    p = pd.DataFrame(x_data[i].T[0].T)
#    #p = p.rename(columns={'1. open': 'Open', '2. high': 'High','3. low': 'Low', '4. close': 'Close'})
#    p.columns = ['Open', 'High', 'Low', 'Close']
#    p.index = test_date_list
#    mpf.plot(p,type='candle', title=str(i))

# =============================================================================
# 
# pred = model_binary.predict(x_validation)
# binary_threshold = pred.mean() + pred.std()
# prediction_binary = pred > binary_threshold
# 
# #    x = x / x.max()
# 
# 
# 
# 
# 
# 
# =============================================================================

































#total_x = np.array(total_df)
