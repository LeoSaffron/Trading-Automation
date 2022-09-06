# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 14:49:43 2020

@author: jonsnow
"""

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
from datetime import datetime, timedelta
from keras.utils import to_categorical


frame_size = 100
#y_interval = 15
#interval = 5
#pattern_count = 1050
#limit_X = 30000
lower_limit = 0.01
upper_limit = 0.1
threshold_Y_binary_percent = 1.030

#path_tickers_list = "./middle_volatility_potential_top100.csv"
path_stock_data = "stock_data/fetched_by_iqfeed"
output_dataframes_folder_initial = "stock_data/filtered_vol_std/"
#market_cap_path = "tickers_with_market_cap.csv"

#marketcap_df = pd.read_csv(market_cap_path).set_index('symbol')

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

#def calculate_10day_average_volume_in_df(df_list, days_to_average_count = 10):
#    avg_days_volume_list_by_dfs = []
#    for df_index in range(len(df_list)):
#        avg_volume_list = []
#        avg_days_volume_list = []
#        for i in range(len(df_list[df_index])):
#            avg_volume_list.append(df_list[df_index][i]['Volume'].sum())
#            if(i < days_to_average_count):
#                avg_days_volume_list.append(None)
#            else:
#                avg_days_volume_list.append(sum(avg_volume_list[i - days_to_average_count:i]) / days_to_average_count)
#        avg_days_volume_list_by_dfs.append(avg_days_volume_list)
#    return avg_days_volume_list_by_dfs
#
#
#def get_df_with_10day_average_volume(df):
#    pass
#
#def get_market_cap_for_df_list(df_list):
#    market_cap_list = []
#    for ticker_index in range(len(df_list)):
#        market_cap = []
#        for day_index in df_list[ticker_index]:
#            try:
#                ticker = df_list[ticker_index][0]['symbol'][0]
#                market_cap.append(marketcap_df.loc[ticker]['market_cap'])
#            except ValueError:
#                print ("error")
#                market_cap.append(None)
#            except:
#                print ("Unexpected error:", sys.exc_info()[0])
#                market_cap.append(None)
#        market_cap_list.append(market_cap)
#    return market_cap_list

def get_tickers_and_dates_indices_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry):
    result_indices = []
#    test_potential_volatility_list = []
    volatility_average_list = calculate_10day_average_volume_in_df_list(df_list, days_to_average_count = 10)
#    market_capitalization_list = get_market_cap_for_df_list(df_list)
    for ticker_index in range(len(df_list)):
        for day_index in range(len(df_list[ticker_index])):
#                vol = volumes_average_list[ticker_index][day_index]
#                cap = market_capitalization_list[ticker_index][day_index]
            daily_volatility = volatility_average_list[ticker_index][day_index]
            if (daily_volatility == None or daily_volatility == 0):
                continue
#            if(not my_is_number(daily_volatility)):
#                continue
#                potential_volalitily = cap / vol
#            stock_std = df_list[ticker_index][day_index].std()
#            potential_volalitily = vol / cap
            
            if(my_is_number(daily_volatility)):
                if ((daily_volatility >= minimum_boundry) and 
                    (daily_volatility <= maximum_boundry)):
                    result_indices.append((ticker_index, day_index, daily_volatility))
#    return result_indices, test_potential_volatility_list
    return result_indices

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value
    
def get_df_by_std_volatility_grouped_by_days(df_list, minimum_boundry, maximum_boundry):
    frame_length = 0
    X_list = []
#    Y_list = []
    list_to_test = get_tickers_and_dates_indices_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry)
    for i in range(len(list_to_test)):
        current_day_frame_length = len(df_list[list_to_test[i][0]][list_to_test[i][1]])
        if(current_day_frame_length < frame_length):
            continue
#        current_df = df_list[list_to_test[i][0]][list_to_test[i][1]].drop('symbol', axis=1)
        current_df = df_list[list_to_test[i][0]][list_to_test[i][1]]
#        y_value = check_df_for_maximum_gain_percent(df_list[list_to_test[i][0]][list_to_test[i][1]], frame_length)
#        if(y_value == float('inf')):
#            continue
        X_list.append(current_df)
#        Y_list.append(check_df_for_maximum_gain_percent(df_list[list_to_test[i][0]][list_to_test[i][1]], frame_length))
#    return X_list, Y_list, test_potential_volatility_list
    return X_list
    
def get_clean_X_Y_percentage_from_df_by_marketcap_volume(df_list, minimum_boundry, maximum_boundry):
    frame_length = 100
    X_list = []
    Y_list = []
    list_to_test = get_tickers_and_dates_indices_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry)
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
    return X_list, Y_list
        

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
#
#def fill_0_volume_rows_in_dflist_by_indices(dflist, ticker_index, df_index):
#    d = dflist[ticker_index][df_index]
#    for i in range(1,len(d)):
#        time_delta = d.index[i] - d.index[i - 1]
#        if(time_delta != timedelta(minutes=1)):
#            #print(i)
#            new_date = d.index[i - 1] + timedelta(minutes=1)
#            #print(new_date)
#            new_price = d.iloc[i - 1]['Close']
#            new_row = {
#            #'date': new_date,
#            'Open': new_price,
#            'Low': new_price,           
#            'High': new_price,
#            'Close': new_price,
#            'Volume' : 0 ,
#            'symbol' : d.iloc[i - 1]['symbol']
#            }
#            #print(new_row)
#            temp_df = pd.DataFrame(new_row, index=[new_date])
#            #temp_df.set_index('date')
#            #print(temp_df)
#           # print(temp_df.index)
#            d = pd.concat([d,temp_df],ignore_index=False)
#            #d.set_index(temp_index_list)
#            #d.loc[len(d.index)] = new_row
#            #d.index[len(d.index) - 1] = new_date
#    dflist[ticker_index][df_index] = d.sort_index()
#

#lower_limit = 1000000
#upper_limit = 6000000
#output_dataframes_folder_initial = "stock_data/filtered_vol/"

def save_files(df_list, output_folder, lower_limit, upper_limit):
    try:
        outdir_temp = output_folder + "l" + str(lower_limit) + '/'
        if not os.path.exists(outdir_temp):
            os.mkdir(outdir_temp)
        outdir_temp = outdir_temp + "u" + str(upper_limit) + '/'
        if not os.path.exists(outdir_temp):
            os.mkdir(outdir_temp)
    except:
        print( "Unexpected error:", sys.exc_info()[0])
    output_dataframes_folder = output_folder + "l" + str(lower_limit) + '/' + "u" + str(upper_limit) + '/'
        
    for df in df_list:
        ticker = df['symbol'][0]
        frame_date = str(df.index[0].date())
        try:
            outdir_temp = output_dataframes_folder  + str(ticker) + '/'
            if not os.path.exists(outdir_temp):
                os.mkdir(outdir_temp)
            output_file_path = outdir_temp + frame_date + ".csv"
            df.to_csv(output_file_path)
        except:
            print( "Unexpected error:", sys.exc_info()[0])
            print(outdir_temp)
            print( "Error in ticker {}",ticker)
#output_dataframes_folder = "stock_data/filtered_vol/"


#def fill_0_volume_rows_in_whole_list(dflist):
#    for i in range(len(dflist)):
#        print("filling {} 0volume".format(i))
#        for j in range(len(dflist[i])):
#            fill_0_volume_rows_in_dflist_by_indices(dflist, i, j)
    
def get_clean_data_df_from_files_grouped_by_day(files_list):
    
    DFList_all = []
    for ticker_file_tuple in files_list:
        path_ticker, ticker_file  = ticker_file_tuple
#        print(path_ticker)
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
#lower_limit = 0.2
#upper_limit = 0.5

def get_X_Y_data_from_filelist(filelist):
    clean_df = get_clean_data_df_from_files_grouped_by_day(filelist)
#    print('filling 0volume rows')
#    fill_0_volume_rows_in_whole_list(clean_df)
    print('getting volume groups df')
#    df_X, df_Y, test_potential_volatility_list = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, 1000000, 6000000)
    df_X = get_df_by_std_volatility_grouped_by_days(clean_df, lower_limit, upper_limit)
#    print(df_X)
#    print('fomatting df')
#    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
#    df_Y_binary = np.array(df_Y)
#    return df_X, df_Y_binary, test_potential_volatility_list
    return df_X

file_batch_size = 40

#output_dataframes_folder_initial = "stock_data/filtered_vol/"
df_all_X = []
df_all_Y_binary = []
test_potential_volatility_list = []
for i in range(int(len(filelist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(filelist_stock_data) / file_batch_size) + 1))
#    df_X, df_Y, vol_temp_list = get_X_Y_data_from_filelist(
#            filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))])
    df_X = get_X_Y_data_from_filelist(
            filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))])
#    for x in vol_temp_list:
#        test_potential_volatility_list.append(x) 
    
    save_files(df_X, output_dataframes_folder_initial, lower_limit, upper_limit)
#    for x in df_X : 
#        df_all_X.append(x) 
#    for y in df_Y : 
#        df_all_Y_binary.append(y) 
#save_files(df_all_X, output_dataframes_folder_initial, lower_limit, upper_limit)

