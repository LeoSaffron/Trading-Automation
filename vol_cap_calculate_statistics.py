# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:06:06 2020

@author: jonsnow
"""

#from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import sys
import mplfinance as mpf
import numpy as np
import pandas as pd

import requests
#from requests.auth import HTTPDigestAuth
import json
from datetime import datetime

#%matplotlib inline
#import pandas_datareader
import datetime
from pandas.plotting import scatter_matrix
import time
#import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import PIL
import tensorflow as tf

from keras import optimizers
import tensorflow.keras.backend
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
#from yahoofinancials import YahooFinancials

import json
from keras.models import model_from_json, load_model
from datetime import date
import os
from sklearn.model_selection import train_test_split
from keras.layers import Dropout


frame_size = 100
#y_interval = 15
#interval = 5
#pattern_count = 1050
#limit_X = 30000
threshold_Y_binary_percent = 1.02

path_tickers_list = "./middle_volatility_potential_top100.csv"
path_stock_data = "stock_data/fetched_by_iqfeed"
market_cap_path = "tickers_with_market_cap.csv"

marketcap_df = pd.read_csv(market_cap_path).set_index('symbol')

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []


def calculate_10day_average_volume_in_df(df_list, days_to_average_count = 10):
    avg_days_volume_list_by_dfs = []
    for df_index in range(len(df_list)):
        avg_volume_list = []
        avg_days_volume_list = []
        for i in range(len(df_list[df_index])):
            avg_volume_list.append(df_list[df_index][i]['Volume'].sum())
            if(i < days_to_average_count):
                avg_days_volume_list.append(None)
            else:
                avg_days_volume_list.append(sum(avg_volume_list[i - days_to_average_count:i]) / days_to_average_count)
        avg_days_volume_list_by_dfs.append(avg_days_volume_list)
    return avg_days_volume_list_by_dfs


def get_df_with_10day_average_volume(df):
    pass

def get_market_cap_for_df_list(df_list):
    market_cap_list = []
    for ticker_index in range(len(df_list)):
        market_cap = []
        for day_index in df_list[ticker_index]:
            try:
                ticker = df_list[ticker_index][0]['symbol'][0]
                market_cap.append(marketcap_df.loc[ticker]['market_cap'])
            except ValueError:
                print ("error")
                market_cap.append(None)
            except:
                print ("Unexpected error:", sys.exc_info()[0])
                market_cap.append(None)
        market_cap_list.append(market_cap)
    return market_cap_list

def get_tickers_and_dates_indices_within_market_volume_bondries(df_list, minimum_boundry, maximum_boundry):
    result_indices = []
    test_potential_volatility_list = []
    volumes_average_list = calculate_10day_average_volume_in_df(df_list, days_to_average_count = 10)
    market_capitalization_list = get_market_cap_for_df_list(df_list)
    for ticker_index in range(len(df_list)):
        for day_index in range(len(df_list[ticker_index])):
                vol = volumes_average_list[ticker_index][day_index]
                cap = market_capitalization_list[ticker_index][day_index]
                if (vol == None):
                    continue
                if (cap ==None):
                    continue
                if (cap == 0):
                    continue
                if((not my_is_number(vol)) or (not my_is_number(cap))):
                    continue
                potential_volalitily = cap / vol
                
                if(my_is_number(potential_volalitily)):
                    test_potential_volatility_list.append(potential_volalitily)
                    if ((potential_volalitily >= minimum_boundry) and 
                        (potential_volalitily <= maximum_boundry)):
                        result_indices.append((ticker_index, day_index, cap / vol))
    return result_indices, test_potential_volatility_list

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value
    
def get_clean_X_Y_percentage_from_df_by_marketcap_volume(df_list, minimum_boundry, maximum_boundry):
    frame_length = 100
    X_list = []
    Y_list = []
    list_to_test, test_potential_volatility_list = get_tickers_and_dates_indices_within_market_volume_bondries(df_list, minimum_boundry, maximum_boundry)
    for i in range(len(list_to_test)):
        current_day_frame_length = len(df_list[list_to_test[i][0]][list_to_test[i][1]])
        if(current_day_frame_length < frame_length):
            continue
        current_df = df_list[list_to_test[i][0]][list_to_test[i][1]][:frame_length].drop('symbol', axis=1)
        y_value = check_df_for_maximum_gain_percent(df_list[list_to_test[i][0]][list_to_test[i][1]], frame_length)
        if(y_value == float('inf')):
            continue
        X_list.append(current_df)
        Y_list.append(check_df_for_maximum_gain_percent(df_list[list_to_test[i][0]][list_to_test[i][1]], frame_length))
    return X_list, Y_list, test_potential_volatility_list
        

def my_is_number(var_to_test):
    try:
        var_to_test / 1
        if(var_to_test > 0 or var_to_test < 1):
            return True
        return False
    except:
        return False


def get_stock_data_filelist_from_folder(path_all):
    df_files = []
    for path_ticker in get_immediate_subdirectories(path_all):
#        print(path_all+"/"+path_ticker)
        current_path = path_all+"/"+path_ticker+"/"
        filelist = [name for name in os.listdir(current_path)]
        mydata = ''
        if(len(filelist) == 0):
            continue
        try:
            df_files.append((path_ticker,current_path + filelist[0]))
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
    return df_files
    
def get_clean_data_df_from_files_grouped_by_day(files_list):
    
    DFList_all = []
    for ticker_file_tuple in files_list:
        path_ticker, ticker_file  = ticker_file_tuple
        print(path_ticker)
        try:
            mydata = pd.read_csv(ticker_file)
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
        mydata = mydata.drop(['Open Interest'], axis=1)
#        print('hallo')
#        mydata = mydata.rename(columns={'Unnamed: 0' : 'date', 'o' : 'Open', 'h': 'High','l': 'Low', 'c': 'Close', 'v' : 'Volume'})
        mydata = mydata.rename(columns={'Date' : 'date'})
#        print('haiduc')
#        mydata.date = pd.to_datetime(mydata.date, unit='date')
        mydata.date = pd.to_datetime(mydata.date)
        mydata['symbol'] = path_ticker
        mydata = mydata.set_index('date')
        DFList = []
#        print('ci pero')
        for groupyear in mydata.groupby(mydata.index.year):
            for groupmonth in groupyear[1].groupby(groupyear[1].index.month):
                for group in groupmonth[1].groupby(groupmonth[1].index.day):
                        DFList.append(group[1])
        DFList_all.append(DFList)
#        print('ubiro')
    return DFList_all
#        

    
#total_df_all = get_df_from_api()
filelist_stock_data = get_stock_data_filelist_from_folder(path_stock_data)

def get_X_Y_data_from_filelist(filelist):
    clean_df = get_clean_data_df_from_files_grouped_by_day(filelist)
    df_X, df_Y, test_potential_volatility_list = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, 0.0001, 0.002)
    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
    return df_X, df_Y_binary, test_potential_volatility_list

file_batch_size = 40
df_all_X = []
df_all_Y_binary = []
test_potential_volatility_list = []
for i in range(int(len(filelist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(filelist_stock_data) / file_batch_size) + 1))
    df_X, df_Y, vol_temp_list = get_X_Y_data_from_filelist(
            filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))])
    for x in vol_temp_list:
        test_potential_volatility_list.append(x) 
    for x in df_X : 
        df_all_X.append(x) 
    for y in df_Y : 
        df_all_Y_binary.append(y) 
test_potential_volatility_list = np.array(test_potential_volatility_list)
test_potential_volatility_list.sort()
vols = np.array_split(test_potential_volatility_list,20)
vols_means = []
for vol in vols:
    vols_means.append(np.array(vol).mean())
boundries = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
last_boundry = 0
next_boundry = boundries[0]
start_index = 0
boundry_index = 0
split_indices = [0]
for i in range(len(test_potential_volatility_list)):
    if(test_potential_volatility_list[i] > next_boundry):
        split_indices.append(i)
        boundry_index += 1
        if(boundry_index >= len(boundries)):
            break
        next_boundry = boundries[boundry_index]
counts = []
for i in range(len(split_indices)):
    if(i >= len(split_indices) - 1):
         counts.append(len(test_potential_volatility_list) - split_indices[i])
    else:
        counts.append(split_indices[i + 1] - split_indices[i])
        
boundries.append(1000)

fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#plt.plot(counts)
#ax.bar(counts,boundries)
plt.bar(x=boundries,height=counts)
#ax.bar(boundries, counts)
plt.show()
#clean_df = get_clean_data_df_from_files_grouped_by_day(filelist_stock_data)
##df_all_X, df_Y = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, 0.015, 0.2)
#df_all_X, df_all_Y = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, 0.01, 0.1)
#df_all_Y_binary = (np.array(df_all_Y) > threshold_Y_binary_percent)
##########################
########## SCALE
##########################



#for df_list in df_all_X:
for i in range(len(df_all_X)):
    high = df_all_X[i].drop(['Volume'], axis=1)['High'].max()
    low = df_all_X[i].drop(['Volume'], axis=1)['Low'].min()
    delta = high - low
    df_all_X[i]['Open'] =  (df_all_X[i]['Open'] - low) / delta
    df_all_X[i]['High'] =  (df_all_X[i]['High'] - low) / delta
    df_all_X[i]['Low'] =  (df_all_X[i]['Low'] - low) / delta
    df_all_X[i]['Close'] =  (df_all_X[i]['Close'] - low) / delta
    
    high_volume = df_all_X[i]['Volume'].max()
    low_volume = df_all_X[i]['Volume'].min()
    delta_volume = high_volume - low_volume
    df_all_X[i]['Volume'] =  (df_all_X[i]['Volume'] - low_volume) / delta_volume
#
#for i in range(len(x_data)):
#    x_data[i] =x_data[i] - x_data[i].min()
#    x_data[i] = x_data[i] / x_data[i].max()


##########################
######## RERRANGE
##########################
for i in range(len(df_all_X)):
    df_all_X[i]['High2'] = df_all_X[i]['High'].copy()
    df_all_X[i]['Close2'] = df_all_X[i]['Close'].copy()
    df_all_X[i]['Open2'] = df_all_X[i]['Open'].copy()
    df_all_X[i]['Low2'] = df_all_X[i]['Low'].copy()
for i in range(len(df_all_X)):
    df_all_X[i] = df_all_X[i].reindex(columns= ['High', 'High2', 'Close', 'Close2', 'Open', 'Open2', 'Low', 'Low2'])
for i in range(len(df_all_X)):
    df_all_X[i].insert(loc=0, column='padding 0', value=np.zeros(df_all_X[i].shape[0]))
    df_all_X[i].insert(loc=9, column='padding 9', value=np.zeros(df_all_X[i].shape[0]))

##########################
##### TRAIN TEST SPLIT
##########################

df_all_X = np.array(df_all_X)
df_all_X = df_all_X.reshape(df_all_X.shape[0], frame_size, 10, 1)
X_train, X_test, Y_train, Y_test = train_test_split(np.array(df_all_X), df_all_Y_binary, test_size=0.20)

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

model_binary = tf.keras.models.Sequential([
# YOUR CODE HERE
        tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape = (frame_size,10,1)),
        # tf.keras.layers.Conv2D(64, (1,3), activation='relu'),
        tf.keras.layers.Conv2D(64, (1,2), activation='relu'),
        tf.keras.layers.Conv2D(32, (1,1), activation='relu'),
        #tf.keras.layers.Conv2D(64, (1,5), activation='relu'),
        tf.keras.layers.Conv2D(16, (2,2), activation='relu'),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        # tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
#       tf.keras.layers.Dense(1)
])

#model_binary.compile( optimizer="adam", loss='binary_crossentropy',  metrics=['accuracy', precision, recall])
model_binary.compile( optimizer="adam", loss='binary_crossentropy',  metrics=['accuracy', precision, recall])
#model_binary.compile( optimizer="adam", loss=precision,  metrics=['accuracy', precision, recall])
#history_binary = model_binary.fit(np.array(X_train),np.array(Y_train),verbose=1,batch_size=32,  epochs=50,validation_data=(X_test,Y_test))
history_binary = model_binary.fit(X_train,np.array(Y_train),verbose=1,batch_size=32,  epochs=100,validation_data=(X_test,np.array(Y_test)))

#test_date_list = pd.to_datetime(pd.read_csv("examples/fetched/ZM/2020_04_16.csv").date[:25])

predictions = model_binary.predict(X_test)

c = 0
for i in range(len(p)):
    if(p[i]):
        if(Y_test[i] == True):
            c += 1
            
true_indices = []
minimum_percent = 0.3
for i in range(len(predictions)):
    if(predictions[i] > minimum_percent):
        true_indices.append(i)
     
items_to_ckeck = [i for i in true_indices if i < limit_X]
print(items_to_ckeck)
for i in items_to_ckeck:
    # p = pd.DataFrame(x_data[true_indices[0]].T[0].T)
    p = pd.DataFrame(x_data[i].T[0].T)
    #p = p.rename(columns={'1. open': 'Open', '2. high': 'High','3. low': 'Low', '4. close': 'Close'})
    p.columns = ['Open', 'High', 'Low', 'Close']
    p.index = test_date_list
    mpf.plot(p,type='candle', title=str(i))



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
