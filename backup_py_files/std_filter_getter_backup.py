# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:16:17 2020

@author: jonsnow
"""

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

#import requests
#from requests.auth import HTTPDigestAuth
#import json
from datetime import datetime, timedelta, date

#%matplotlib inline
#import pandas_datareader
#import datetime
from pandas.plotting import scatter_matrix
#import time
#import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
from PIL import Image
#import PIL
import tensorflow as tf

from keras import optimizers
import tensorflow.keras.backend
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
#from yahoofinancials import YahooFinancials

#import json
#from keras.models import model_from_json, load_model
#from datetime import date
import os
#from sklearn.model_selection import train_test_split
from keras.layers import Dropout
#from datetime import datetime, timedelta
from keras.utils import to_categorical
import pandas_market_calendars as mcal


frame_size = 100
lower_limit = 0.01
upper_limit = 0.1
#threshold_Y_binary_percent = 1.030
path_stock_data = "stock_data/iqfeed_last_months2"
output_dataframes_folder_initial = "stock_data/filtered_vol_std/"
date_to_check = "2020-10-16"


start_date = (datetime.strptime(date_to_check, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
end_date = (datetime.strptime(date_to_check, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

def get_dates_list_to_check_for_date_range(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    
    dates_to_check = nyse.schedule(start_date=start_date, end_date=end_date)
    dates_to_check = mcal.date_range(dates_to_check, frequency='1D')
    dates_to_check_str = []
    for date_in_list in dates_to_check:
        dates_to_check_str.append(str(date_in_list).split(" ")[0])
    return dates_to_check_str


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

def get_tickers_and_dates_indices_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry):
    result_indices = []
    volatility_average_list = calculate_10day_average_volume_in_df_list(df_list, days_to_average_count = 10)
    for ticker_index in range(len(df_list)):
        for day_index in range(len(df_list[ticker_index])):
            daily_volatility = volatility_average_list[ticker_index][day_index]
            if (daily_volatility == None or daily_volatility == 0):
                continue
            
            if(my_is_number(daily_volatility)):
                if ((daily_volatility >= minimum_boundry) and 
                    (daily_volatility <= maximum_boundry)):
                    result_indices.append((ticker_index, day_index, daily_volatility))
    return result_indices

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value
    
def get_df_by_std_volatility_grouped_by_days(df_list, minimum_boundry, maximum_boundry):
    frame_length = 0
    X_list = []
    list_to_test = get_tickers_and_dates_indices_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry)
    for i in range(len(list_to_test)):
        current_day_frame_length = len(df_list[list_to_test[i][0]][list_to_test[i][1]])
        if(current_day_frame_length < frame_length):
            continue
        current_df = df_list[list_to_test[i][0]][list_to_test[i][1]]
        X_list.append(current_df)
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

def get_stock_data_filelist_from_folder(path_all, date_list):
    df_files = []
    filelist = []
    for x in date_list:
        filelist.append("{}.csv".format(x))
    for path_ticker in get_immediate_subdirectories(path_all):
        current_path = path_all+"/"+path_ticker+"/"
#        filelist = [name for name in os.listdir(current_path)]
        tmp_list = []
        if(len(filelist) == 0):
            continue
        try:
            for filename in filelist:
                tmp_list.append((path_ticker,current_path + filename))
#                df_files.append((path_ticker,current_path + filename))
            df_files.append(tmp_list)
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
        
    return df_files

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
dates_to_download = get_dates_list_to_check_for_date_range(start_date, end_date)
filelist_stock_data = get_stock_data_filelist_from_folder(path_stock_data, dates_to_download)[:190]
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

