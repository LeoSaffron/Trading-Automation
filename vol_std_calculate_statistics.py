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


#frame_size = 100
#y_interval = 15
#interval = 5
#pattern_count = 1050
#limit_X = 30000
#threshold_Y_binary_percent = 1.02

path_stock_data = "stock_data/fetched_by_iqfeed"


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []


def calculate_sindlge_day_std_df(df):
    return df.drop(['Volume', 'symbol'], axis=1).std().mean()

def calculate_10day_average_volume_in_df_list(df_list, days_to_average_count = 10):
    std_list_by_dfs = []
    for df_index in range(len(df_list)):
        std_single_day_list = []
        std_single_stock_list = []
        for i in range(len(df_list[df_index])):
            std_single_day_list.append(calculate_sindlge_day_std_df(df_list[df_index][i]))
            if(i < days_to_average_count):
                std_single_stock_list.append(None)
            else:
                std_single_stock_list.append(sum(std_single_day_list[i - days_to_average_count:i]) / days_to_average_count )
        std_list_by_dfs.append(std_single_stock_list)
    return std_list_by_dfs

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

filelist_stock_data = get_stock_data_filelist_from_folder(path_stock_data)

def get_std_data_from_filelist(filelist):
    clean_df = get_clean_data_df_from_files_grouped_by_day(filelist)
    df_X, df_Y, test_potential_volatility_list = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, 0.0001, 0.002)
    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
    std_list_local = calculate_10day_average_volume_in_df_list(clean_df)
    return std_list_local


file_batch_size = 40
df_all_X = []
df_all_Y_binary = []
test_potential_volatility_list = []
std_list = []
for i in range(int(len(filelist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(filelist_stock_data) / file_batch_size) + 1))
    std_list_local = get_std_data_from_filelist(
                    filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))]
                    )
    for x in std_list_local : 
        std_list.append(x) 

std_list_flat = []
for lst in std_list:
    try:
        for x in lst : 
            if(my_is_number(x)):
                std_list_flat.append(x)
    except KeyboardInterrupt:
        break
    except:
        pass
#        continue
        
std_list_flat = np.array(std_list_flat)
std_list_flat.sort()
vols = np.array_split(std_list_flat,500)
vols_means = []
for vol in vols:
    vols_means.append(np.array(vol).mean())
#boundries = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
#
##boundries = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
#boundries = list(np.linspace(0.05,0.5,100))
#last_boundry = 0
#next_boundry = boundries[0]
#start_index = 0
#boundry_index = 0
#split_indices = [0]
#for i in range(len(std_list_flat)):
#    if(std_list_flat[i] > next_boundry):
#        split_indices.append(i)
#        boundry_index += 1
#        if(boundry_index >= len(boundries)):
#            break
#        next_boundry = boundries[boundry_index]
#counts = []
#for i in range(len(split_indices)):
#    if(i >= len(split_indices) - 1):
#         counts.append(len(std_list_flat) - split_indices[i])
#    else:
#        counts.append(split_indices[i + 1] - split_indices[i])
#        
#boundries.append(2)
#
#fig = plt.figure()
##ax = fig.add_axes([0,0,1,1])
##plt.plot(counts)
##ax.bar(counts,boundries)
#plt.bar(x=boundries,height=counts)
##ax.bar(boundries, counts)
#plt.show()
#
#
#
#
#vols2 = np.array_split(std_list_flat,100000)
#vols2_means = []
#for vol2 in vols2:
#    vols2_means.append(np.array(vol2).mean())
#    
#
#
#boundries = list(np.linspace(0.005,2.5,1000))
#last_boundry = 0
#next_boundry = boundries[0]
#start_index = 0
#boundry_index = 0
#split_indices = [0]
#for i in range(len(vols2_means)):
#    if(vols2_means[i] > next_boundry):
#        split_indices.append(i)
#        boundry_index += 1
#        if(boundry_index >= len(boundries)):
#            break
#        next_boundry = boundries[boundry_index]
#counts = []
#for i in range(len(split_indices)):
#    if(i >= len(split_indices) - 1):
#         counts.append(len(vols2_means) - split_indices[i])
#    else:
#        counts.append(split_indices[i + 1] - split_indices[i])
#        
#boundries.append(3)
#
#l = 4
#h = 200
#
#fig = plt.figure()
##ax = fig.add_axes([0,0,1,1])
##plt.plot(counts)
##ax.bar(counts,boundries)
##plt.bar(x=boundries,height=counts)
#plt.plot(boundries[l:h],counts[l:h])
##ax.bar(boundries, counts)
#plt.show()

median = std_list_flat[int(len(std_list_flat) / 2)]

fig, ax = plt.subplots() 
ax.hist(std_list_flat, bins = np.linspace(0.18,0.6,100))


#
#test_potential_volatility_list = np.array(test_potential_volatility_list)
#test_potential_volatility_list.sort()
#vols = np.array_split(test_potential_volatility_list,20)
#vols_means = []
#for vol in vols:
#    vols_means.append(np.array(vol).mean())
#boundries = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
#last_boundry = 0
#next_boundry = boundries[0]
#start_index = 0
#boundry_index = 0
#split_indices = [0]
#for i in range(len(test_potential_volatility_list)):
#    if(test_potential_volatility_list[i] > next_boundry):
#        split_indices.append(i)
#        boundry_index += 1
#        if(boundry_index >= len(boundries)):
#            break
#        next_boundry = boundries[boundry_index]
#counts = []
#for i in range(len(split_indices)):
#    if(i >= len(split_indices) - 1):
#         counts.append(len(test_potential_volatility_list) - split_indices[i])
#    else:
#        counts.append(split_indices[i + 1] - split_indices[i])
#        
#boundries.append(1000)