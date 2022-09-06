#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:32:49 2020

@author: napol
"""

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

import pandas_market_calendars as mcal
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

path_stock_data = "stock_data/stocks_2019_2020"
#path_stock_data = "stock_data/filtered_vol_std/l0.9/u2.9"
lower_limit = 0.3
upper_limit = 3.0
output_dataframes_folder_initial = "stock_data/std_filtered/"
#output_dataframes_folder_initial = "/home/napol/cached_stocks/std_filtered/"
output_folder = output_dataframes_folder_initial
output_dataframes_folder = output_folder + "l" + str(lower_limit) + '/' + "u" + str(upper_limit) + '/'


start_date='2020-09-21'
end_date='2020-11-25'


try:
    outdir_temp = output_folder + "l" + str(lower_limit) + '/'
    if not os.path.exists(outdir_temp):
        os.mkdir(outdir_temp)
    outdir_temp = outdir_temp + "u" + str(upper_limit) + '/'
    if not os.path.exists(outdir_temp):
        os.mkdir(outdir_temp)
except:
    print( "Unexpected error:", sys.exc_info()[0])

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
    return maximum_value / close_value

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
#    max_gain_list_by_dfs = []
    for df_index in range(len(df_list)):
        std_single_day_list = []
        std_single_stock_maxgain_combines_list = []
#        max_gain_day_list = []
        for i in range(len(df_list[df_index])):
            std_single_day_list.append(calculate_sindlge_day_std_df(df_list[df_index][i]))
#            max_gain_day_list.append(check_df_for_maximum_gain_percent(df_list[df_index][i], frame_size))
            std_single_stock_variable = None
            max_gain_this_day = None
            if(len(df_list[df_index][i]) > frame_size):
                max_gain_this_day = check_df_for_maximum_gain_percent(df_list[df_index][i], frame_size)
#            if(i < days_to_average_count):
#                std_single_stock_list.append(None)
            if(i >= days_to_average_count):
                std_single_stock_variable = sum(std_single_day_list[i - days_to_average_count:i]) / days_to_average_count 
                if(std_single_stock_variable >= lower_limit and std_single_stock_variable <= upper_limit):
#                    print("posititve {} {} ".format(df_index, i))
                    save_df_by_index(df_list, df_index, i)
            std_single_stock_maxgain_combines_list.append((std_single_stock_variable, max_gain_this_day, df_list[df_index][i]['High'][0]))
            
        std_maxgain_pair_list_by_dfs.append(std_single_stock_maxgain_combines_list)
#        max_gain_list_by_dfs.append(max_gain_day_list)
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
    
def get_clean_data_df_from_folder_list_grouped_by_day(folder_list):
    
    DFList_all = []
    for ticker_file_tuple in folder_list:
        path_ticker, ticker_folder  = ticker_file_tuple
        
        filelist = [name for name in os.listdir(ticker_folder)]
        filelist = [name for name in filelist if name in valid_filenames]
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
#            mydata = mydata.rename(columns={'Date' : 'date'})
#            mydata.date = pd.to_datetime(mydata.date)
            mydata.Date = pd.to_datetime(mydata.Date)
            mydata['symbol'] = path_ticker
#            mydata = mydata.set_index('date')
            mydata = mydata.set_index('Date')
            
            DFList.append(mydata)
        DFList_all.append(DFList)
    return DFList_all

def save_df_by_index(df_list, ticker_index, day_index):
    ticker = df_list[ticker_index][day_index]['symbol'][0]
    current_output_path = output_dataframes_folder + ticker + '/'
    if not os.path.exists(current_output_path):
        os.mkdir(current_output_path)
    current_date = str(df_list[ticker_index][day_index].index[0]).split(" ")[0]
    destination_file_path = current_output_path + current_date + ".csv"
    df_list[ticker_index][day_index].to_csv(destination_file_path)

#def get_clean_data_df_from_files_grouped_by_day(files_list):
#    
#    DFList_all = []
#    for ticker_file_tuple in files_list:
#        path_ticker, ticker_file  = ticker_file_tuple
##        print(path_ticker)
#        try:
#            mydata = pd.read_csv(ticker_file)
#        except KeyboardInterrupt:
#            break
#        except:
#            print("An exception occurred with ticker {} ".format(path_ticker))
#            faulty_tickers.append(path_ticker)
#            continue
#        try:
#            mydata = mydata.drop(['Open Interest'], axis=1)
#        except KeyboardInterrupt:
#            break
#        except:
#            pass
##        print('hallo')
##        mydata = mydata.rename(columns={'Unnamed: 0' : 'date', 'o' : 'Open', 'h': 'High','l': 'Low', 'c': 'Close', 'v' : 'Volume'})
#        mydata = mydata.rename(columns={'Date' : 'date'})
##        print('haiduc')
##        mydata.date = pd.to_datetime(mydata.date, unit='date')
#        mydata.date = pd.to_datetime(mydata.date)
#        mydata['symbol'] = path_ticker
#        mydata = mydata.set_index('date')
#        DFList = []
##        print('ci pero')
#        for groupyear in mydata.groupby(mydata.index.year):
#            for groupmonth in groupyear[1].groupby(groupyear[1].index.month):
#                for group in groupmonth[1].groupby(groupmonth[1].index.day):
#                        DFList.append(group[1])
#        DFList_all.append(DFList)
##        print('ubiro')
#    return DFList_all


# Create a calendar
start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=14)).strftime('%Y-%m-%d')

def get_dates_list_to_check_for_date_range(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    
    dates_to_check = nyse.schedule(start_date=start_date, end_date=end_date)
    dates_to_check = mcal.date_range(dates_to_check, frequency='1D')
    dates_to_check_str = []
    for date_in_list in dates_to_check:
        dates_to_check_str.append(str(date_in_list).split(" ")[0])
    return dates_to_check_str
dates_to_check = get_dates_list_to_check_for_date_range(start_date, end_date)

valid_filenames = []
for x in dates_to_check:
    valid_filenames.append(x + '.csv')


filelist_stock_data = get_stock_data_filelist_from_folder(path_stock_data)
#
#for i in range(len(filelist_stock_data)):
#    if (filelist_stock_data[i][0] == 'ESGR'):
#        print(i)

def get_std_data_from_filelist(filelist):
#    clean_df = get_clean_data_df_from_files_grouped_by_day(filelist)
    clean_df = get_clean_data_df_from_folder_list_grouped_by_day(filelist)
#    get_clean_data_df_from_folder_list_grouped_by_day
#    df_X, df_Y, test_potential_volatility_list = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, 0.0001, 0.002)
#    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
    std_maxgain_list_local = calculate_10day_average_volatility_and_maxgain_in_df_list(clean_df, days_to_average_count = 10, frame_size = 100)
    return std_maxgain_list_local


file_batch_size = 40
minimum_price = 7
df_all_X = []
df_all_Y_binary = []
test_potential_volatility_list = []
std_maxgain_pair_list = []
for i in range(int(len(filelist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(filelist_stock_data) / file_batch_size) + 1))
    std_list_local = get_std_data_from_filelist(
                    filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))]
                    )
    for x in std_list_local : 
        std_maxgain_pair_list.append(x) 

std_maxgain_pair_list_flat = []
for lst in std_maxgain_pair_list:
    for x in lst : 
        try:
            if(my_is_number(x[0]) and my_is_number(x[1])):
                if(x[2] > minimum_price):
                    std_maxgain_pair_list_flat.append((x[0], x[1]))
        except KeyboardInterrupt:
            break
        except:
            pass
#        continue
plt.scatter(*zip(*std_maxgain_pair_list_flat))
plt.show()        

std_maxgain_pair_list_flat = np.array(std_maxgain_pair_list_flat)
std_maxgain_pair_list_flat = np.array(pd.DataFrame(std_maxgain_pair_list_flat).sort_values(by=0))
x = std_maxgain_pair_list_flat.T[0]
y = std_maxgain_pair_list_flat.T[1]
plt.xlim([0, 5])
plt.plot(x,y)

threshold = 1.02
percantage_achieve_list = []
for x in std_maxgain_pair_list_flat:
    percantage_achieve_list.append((x[0], x[1] > threshold))
percantage_achieve_list = list(percantage_achieve_list)
percantage_achieve_list.sort(key=lambda x: x[0])

#std_maxgain_pair_list_flat = np.array(std_maxgain_pair_list_flat)
#std_maxgain_pair_list_flat.sort()
#vols = np.array_split(std_maxgain_pair_list_flat,500)
#vols_means = []
#for vol in vols:
#    vols_means.append(np.array(vol).mean())
#boundries = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
#
##boundries = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
    
#boundries = list(np.linspace(0.00,0.5,100))

threshold = 1.02    
def get_lists_rate_by_threshold(threshold = 1.02):
    percantage_achieve_list = []
    for x in std_maxgain_pair_list_flat:
        percantage_achieve_list.append((x[0], x[1] > threshold))
    percantage_achieve_list = list(percantage_achieve_list)
    percantage_achieve_list.sort(key=lambda x: x[0])
    
    boundries = list(np.linspace(0.00,6,50))
    last_boundry = 0
    next_boundry = boundries[0]
    start_index = 0
    boundry_index = 0
    split_indices = [0]
    for i in range(len(percantage_achieve_list)):
        if(percantage_achieve_list[i][0] > next_boundry):
            split_indices.append(i)
            boundry_index += 1
            if(boundry_index >= len(boundries)):
                break
            next_boundry = boundries[boundry_index]
    counts = []
    positives = []
    for i in range(len(split_indices)):
    #    print(i)
        if(i >= len(split_indices) - 1):
    #        print("a")
            counts.append(len(percantage_achieve_list) - split_indices[i])
            positives.append(np.array(percantage_achieve_list[split_indices[i]:]).T[1].sum())
        else:
    #        print("b")
            current_count = split_indices[i + 1] - split_indices[i]
            counts.append(current_count)
            current_positive = 0
            if(current_count > 0):
                current_positive = np.array(percantage_achieve_list[split_indices[i]:split_indices[i + 1]]).T[1].sum()
            positives.append(current_positive)
    success_rate = np.array(positives) / np.array(counts)
    return boundries, counts, positives, success_rate
#
#boundries = list(np.linspace(0.00,0.5,1000))
#last_boundry = 0
#next_boundry = boundries[0]
#start_index = 0
#boundry_index = 0
#split_indices = [0]
positives = []
#positive_current = 0
#totals = []
#total_current = 0
#for i in range(len(percantage_achieve_list)):
#    if(percantage_achieve_list[i][0] > next_boundry):
#        split_indices.append(i)
#        positives.append(positive_current)
#        totals.append(total_current)
#        boundry_index += 1
#        total_current = 0
#        positive_current = 0
#        if(boundry_index < len(boundries)):
#            
#            next_boundry = boundries[boundry_index]
#    total_current += 1
#    if(percantage_achieve_list[i][1]):
#        positive_current += 1
        

bound, count, pos, success = get_lists_rate_by_threshold(threshold = 1.02)


#success_rate = np.array(positives) / np.array(counts)

plt.plot(bound, count[:-1])
plt.plot(bound, pos[:-1])

plt.plot(bound, success[:-1])



#plt.plot(boundries, counts[:-1])
#plt.plot(boundries, positives[:-1])
#
#plt.plot(boundries, success_rate[:-1])
#        
#boundries.append(2)
#
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#plt.plot(counts)
#ax.bar(counts,boundries)
plt.bar(x=boundries,height=counts)
#ax.bar(boundries, counts)
plt.show()

output_path = "measured_stats/volatility_max_gains/20200710/"
pd.DataFrame(bound).to_csv(output_path + "boundries.csv")
pd.DataFrame(count).to_csv(output_path + "counts.csv")
pd.DataFrame(positives).to_csv(output_path + "positives.csv")
pd.DataFrame(success).to_csv(output_path + "success_rate.csv")

df = pd.DataFrame([bound, count, positives, success]).T
df.columns = ['boundries', 'counts', 'positives', 'success_rate']
df.to_csv(output_path + "combined_data.csv")

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