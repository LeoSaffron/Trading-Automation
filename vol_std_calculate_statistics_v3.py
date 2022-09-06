# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:06:06 2020

@author: jonsnow
"""

import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

import requests
import json
from datetime import datetime

#%matplotlib inline
import datetime
from pandas.plotting import scatter_matrix
import time
from PIL import Image
import PIL
import tensorflow as tf

from keras import optimizers
import tensorflow.keras.backend
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image

import json
from keras.models import model_from_json, load_model
from datetime import date
import os
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import pandas_market_calendars as mcal


#frame_size = 100
#y_interval = 15
#interval = 5
#pattern_count = 1050
#limit_X = 30000
#threshold_Y_binary_percent = 1.02

path_stock_data = "stock_data/stocks_2019_2020"


start_date='2019-08-01'
end_date='2021-02-04'

initial_filter_lower = 1.2
initial_filter_higher = 10
initial_filter_lower = 0
initial_filter_higher = 10000


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []


def calculate_sindlge_day_std_df(df):
    return df.drop(['Volume', 'symbol'], axis=1).std().mean()

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    maximum_position = df_scan.index.get_loc(df_scan['High'].idxmax())
    minimum_before_max = df_scan[:maximum_position]['Low'].min() / close_value
    minimum_value = df_scan['Low'].min() / close_value
    return maximum_value / close_value, minimum_before_max, minimum_value

def calculate_10day_average_volatility_in_df_list(df_list, days_to_average_count = 10):
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

def calculate_10day_average_volatility_and_maxgain_in_df_list(df_list, days_to_average_count = 10, frame_size = 100):
    std_maxgain_pair_list_by_dfs = []
    for df_index in range(len(df_list)):
        std_single_day_list = []
        std_single_stock_maxgain_combines_list = []
        for i in range(len(df_list[df_index])):
            std_single_day_list.append(calculate_sindlge_day_std_df(df_list[df_index][i]))
            std_single_stock_variable = None
            max_gain_this_day = None
            min_before_max_gain_this_day = None
            min_percent_this_day = None
            division_close_price = None
            if(len(df_list[df_index][i]) > frame_size):
                max_gain_this_day_tuple = check_df_for_maximum_gain_percent(df_list[df_index][i], frame_size)
                max_gain_this_day = max_gain_this_day_tuple[0]
                min_before_max_gain_this_day = max_gain_this_day_tuple[1]
                min_percent_this_day = max_gain_this_day_tuple[2]
                division_close_price = float(df_list[df_index][i]['Close'].iloc[frame_size-1])
            if(i >= days_to_average_count):
                std_single_stock_variable = sum(std_single_day_list[i - days_to_average_count:i]) / days_to_average_count 
            
            sharpe_df = get_sharpe_for_sub_df_daily(df_list[df_index][i])
            last_price = float(df_list[df_index][i][-1:]['Close'])
            volume_price =  df_list[df_index][i][:frame_size]['Volume'].sum()
            last_volumes_sum =  df_list[df_index][i][frame_size - 15:frame_size]['Volume'].sum()
            df_symbol = df_list[df_index][i]['symbol'].iloc[0]
            df_date = str(df_list[df_index][i].index[0]).split(' ')[0]
            std_single_stock_maxgain_combines_list.append(((df_index, i),df_symbol, df_date, std_single_stock_variable, sharpe_df,
                                                           max_gain_this_day, min_before_max_gain_this_day, min_percent_this_day
                                                           , df_list[df_index][i]['High'][0],
                                                           division_close_price, last_price, volume_price, last_volumes_sum))
            
        std_maxgain_pair_list_by_dfs.append(std_single_stock_maxgain_combines_list)
    return std_maxgain_pair_list_by_dfs

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
            df_files.append((path_ticker,current_path))
        except KeyboardInterrupt:
            break
        except:
            print("An exception occurred with ticker {} ".format(path_ticker))
            faulty_tickers.append(path_ticker)
            continue
    return df_files
    
def get_clean_data_df_from_folder_list_grouped_by_day(folder_list, dates):
    DFList_all = []
    date_file_names = []
    for item in dates:
        date_file_names.append(item + '.csv')
    for ticker_file_tuple in folder_list:
        path_ticker, ticker_folder  = ticker_file_tuple
        
        filelist = [name for name in os.listdir(ticker_folder)]
        filelist = [filename_current for filename_current in date_file_names if filename_current in filelist]
        DFList = []
        if(len(filelist) == 0):
            continue
        for file_path in filelist:
            try:
                mydata = pd.read_csv(ticker_folder +file_path)
            except KeyboardInterrupt:
                break
            except:
                print("An exception occurred with ticker {} ".format(path_ticker))
                faulty_tickers.append(path_ticker)
                continue
            try:
                mydata = mydata.drop(['Open Interest'], axis=1)
            except KeyboardInterrupt:
                break
            except:
                pass
            mydata = mydata.rename(columns={'Date' : 'date'})
            mydata.date = pd.to_datetime(mydata.date)
            mydata['symbol'] = path_ticker
            mydata = mydata.set_index('date')
            
            if((mydata['High'].iloc[-1] >= initial_filter_lower) and (mydata['Low'].iloc[-1] <= initial_filter_higher)):
                DFList.append(mydata)
        DFList_all.append(DFList)
    return DFList_all


def get_sharpe_for_sub_df_daily(df):
    std_df = df.drop(['Volume', 'symbol'], axis=1).std().mean()
    gains_lst = []
    for i in range(1,len(df)):
        gains_lst.append(df['Close'].iloc[i] / df['Close'].iloc[i-1])
    gains_lst = np.array(gains_lst)
    return (gains_lst.mean() - 1) / std_df



def get_dates_list_to_check_for_date_range(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    
    dates_to_check = nyse.schedule(start_date=start_date, end_date=end_date)
    dates_to_check = mcal.date_range(dates_to_check, frequency='1D')
    dates_to_check_str = []
    for date_in_list in dates_to_check:
        dates_to_check_str.append(str(date_in_list).split(" ")[0])
    return dates_to_check_str

start_date_initial = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=14)).strftime('%Y-%m-%d')
dates_to_check = get_dates_list_to_check_for_date_range(start_date_initial, end_date)

filelist_stock_data = get_stock_data_filelist_from_folder(path_stock_data)

def get_std_data_from_filelist(filelist, dates):
    clean_df = get_clean_data_df_from_folder_list_grouped_by_day(filelist, dates)
    std_maxgain_list_local = calculate_10day_average_volatility_and_maxgain_in_df_list(clean_df,
                                                                                       days_to_average_count = 10,
                                                                                       frame_size = 100)
    return clean_df, std_maxgain_list_local


file_batch_size = 40
minimum_price = 7
df_all_X = []
df_all_Y_binary = []
test_potential_volatility_list = []
std_maxgain_pair_list = []
for i in range(int(len(filelist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(filelist_stock_data) / file_batch_size) + 1))
    df_x, std_list_local = get_std_data_from_filelist(
                    filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))],
                    dates_to_check
                    )
    for x in df_x: 
        df_all_X.append(x)
    for x in std_list_local: 
        std_maxgain_pair_list.append(x) 

start_time_datetime = datetime.datetime.strptime("09:30:00", '%H:%M:%S')

minute_to_check_time_past = 30
for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        time_past_at_index_n = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past):
            current_time_str = str(df_all_X[i][j].index[minute_to_check_time_past - 1]).split(' ')[1]
            time_past_at_index_n = ((datetime.datetime.strptime(current_time_str, '%H:%M:%S') - start_time_datetime) / 60).seconds
        std_maxgain_pair_list[i][j] = list(std_maxgain_pair_list[i][j])
        std_maxgain_pair_list[i][j].append(time_past_at_index_n)

for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        std_30 = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past):
            std_30 = df_all_X[i][j][:30].drop(['Volume', 'symbol'], axis=1).std().mean()
        std_maxgain_pair_list[i][j].append(std_30)
        
for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        avg_gain_30 = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past):
            avg_gain_30 = (df_all_X[i][j]['Close'][:30] / df_all_X[i][j]['Close'][:30].shift(periods=1))[1:].mean()
        std_maxgain_pair_list[i][j].append(avg_gain_30)
        
for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        slope_30 = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past):
            slope_30 = (df_all_X[i][j]['Close'].iloc[30 - 1] / df_all_X[i][j]['Close'].iloc[0])
        std_maxgain_pair_list[i][j].append(slope_30)


for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        end_gain_30_45 = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past + 15):
            end_gain_30_45 = (df_all_X[i][j]['Close'].iloc[45 - 1] / df_all_X[i][j]['Close'].iloc[30 - 1])
        std_maxgain_pair_list[i][j].append(end_gain_30_45)


for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        max_gain_30_45 = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past + 15):
            max_gain_30_45 = df_all_X[i][j]['Close'][30:45].max() / df_all_X[i][j]['Close'].iloc[30 - 1]
        std_maxgain_pair_list[i][j].append(max_gain_30_45)
        
for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        volume30 = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past):
            volume30 = df_all_X[i][j]['Volume'][:30].sum()
        std_maxgain_pair_list[i][j].append(volume30)
        

for i in range(len(df_all_X)):
    for j in range(len(df_all_X[i])):
        volume_eod = None
        if (len(df_all_X[i][j]) > minute_to_check_time_past):
            volume_eod = df_all_X[i][j]['Volume'].iloc[-1]
        std_maxgain_pair_list[i][j].append(volume_eod)



std_maxgain_pair_list_flat = []
for lst in std_maxgain_pair_list:
    try:
        for x in lst :
            std_maxgain_pair_list_flat.append(x)
    except KeyboardInterrupt:
        break
    except:
        pass
 


std_maxgain_pair_dataframe_flat = pd.DataFrame(std_maxgain_pair_list_flat)
std_maxgain_pair_dataframe_flat.columns=['indices', 'symbol', 'date', 'std', 'sharpe', 'percent_gain',
                                         'min_before_max', 'minimum',
                                         'max_price', 'division_price', 'last_price',
                                         'volume_price', 'last_volumes',
                                         'minute_at_30', 'std30', 'avg_gain_30',
                                         'slope30', 'eodgain_30_45', 'maxgain30_45', 'volume30', 'volume_eod']

std_maxgain_pair_dataframe_flat['mansharpe30'] = (1 - std_maxgain_pair_dataframe_flat['slope30']) / std_maxgain_pair_dataframe_flat['std30']
std_maxgain_pair_dataframe_flat['mysharpe30'] = (1 - std_maxgain_pair_dataframe_flat['avg_gain_30']) / std_maxgain_pair_dataframe_flat['std30']

std_maxgain_pair_dataframe_flat['last_volumes_new'] = std_maxgain_pair_dataframe_flat['last_volumes'] / std_maxgain_pair_dataframe_flat['max_price']

std_maxgain_pair_dataframe_flat['date'] =  pd.to_datetime(std_maxgain_pair_dataframe_flat['date'], format='%Y-%m-%d')
std_maxgain_pair_dataframe_flat['date'] > datetime.datetime.strptime("2020-07-01", '%Y-%m-%d')
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat.drop(['indices', 'symbol', 'date'], axis=1)
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['division_price'] > 0.5]  
std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
                                                                  ['division_price'] > 2]  
std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
                                                                  ['division_price'] < 5]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['sharpe'] > 0.001]  
std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
                                                                  ['minute_at_30'] < 31]  
std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
                                                                  ['volume30'] > 500000]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['slope30'] > 1.08]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['mansharpe30'] > 0]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['mansharpe30'] < -1 ]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['min_before_max'] >0.98]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['minimum'] >0.995]  
#std_maxgain_pair_dataframe_flat = std_maxgain_pair_dataframe_flat[std_maxgain_pair_dataframe_flat
#                                                                  ['last_volumes_new'] > 469000]  


std_maxgain_pair_dataframe_flat.dropna(subset=['sharpe', 'percent_gain',
                                         'min_before_max', 'minimum',
                                         'max_price', 'division_price', 'last_price',
                                         'volume_price', 'last_volumes', 'volume_eod'], inplace=True)
std_maxgain_pair_dataframe_flat['eod_gain'] = std_maxgain_pair_dataframe_flat['last_price'] / std_maxgain_pair_dataframe_flat['division_price']

def get_means_of_2_parameters(x_column, y_column, division_amount):
    std_maxgain_pair_dataframe_flat_temp = std_maxgain_pair_dataframe_flat.sort_values(by=[x_column])
    indices_split = np.linspace(0,len(std_maxgain_pair_dataframe_flat_temp), division_amount, dtype=int)
    sub_df = []
    for i  in range(division_amount - 1):
        sub_df.append(std_maxgain_pair_dataframe_flat_temp[indices_split[i]: indices_split[i+1]])
    x_means = []
    y_means = []
    for x in sub_df:
        x_means.append(x.mean()[x_column])
        y_means.append(x.mean()[y_column])
    return x_means, y_means

x,y = get_means_of_2_parameters('volume_eod', 'eod_gain', 50)
plt.plot(x[20:],y[20:])


plt.plot(x[20:])
print()
plt.plot(y[20:])
print()

std_maxgain_pair_dataframe_flat['passed_3'] = std_maxgain_pair_dataframe_flat['maxgain30_45'] > 1.02
x,y = get_means_of_2_parameters('std', 'passed_3', 25)
plt.plot(x,y)


sl=0.97
gain_sl = []
for i in range(len(std_maxgain_pair_dataframe_flat)):
    line = std_maxgain_pair_dataframe_flat.iloc[i]
    if (line['minimum'] < sl):
        gain_sl.append(sl)
    else:
        gain_sl.append(line['eod_gain'])
std_maxgain_pair_dataframe_flat['gain_sl'] = gain_sl

x,y = get_means_of_2_parameters('mansharpe30', 'gain_sl', 25)
plt.plot(x,y)


#output_path = "measured_stats/volatility_max_gains/20200710/"
#pd.DataFrame(bound).to_csv(output_path + "boundries.csv")
#pd.DataFrame(count).to_csv(output_path + "counts.csv")
#pd.DataFrame(positives).to_csv(output_path + "positives.csv")
#pd.DataFrame(success).to_csv(output_path + "success_rate.csv")
#
#df = pd.DataFrame([bound, count, positives, success]).T
#df.columns = ['boundries', 'counts', 'positives', 'success_rate']
#df.to_csv(output_path + "combined_data.csv")
