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
import shutil, os


path_stock_data = "stock_data/stocks_2019_2020"
output_path = "stock_data/stonks"


start_date='2020-12-05'
end_date='2021-02-04'



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []

try:
    outdir_temp = output_path + '/'
    if not os.path.exists(outdir_temp):
        os.mkdir(outdir_temp)
except:
    print( "Unexpected error:", sys.exc_info()[0])

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
    
def copy_files_for_dates(folder_list, output_path, dates):
    date_file_names = []
    for item in dates:
        date_file_names.append(item + '.csv')
    for ticker_file_tuple in folder_list:
        path_ticker, ticker_folder  = ticker_file_tuple
        print(path_ticker)
        
        filelist = [name for name in os.listdir(ticker_folder)]
        filelist = [filename_current for filename_current in date_file_names if filename_current in filelist]
        if(len(filelist) == 0):
            continue
        outdir_temp = output_path + '/' + path_ticker
        try:
            if not os.path.exists(outdir_temp):
                os.mkdir(outdir_temp)
        except:
            print( "Unexpected error:", sys.exc_info()[0])
        for file_path in filelist:
            output_file = outdir_temp + '/' + file_path
            try:
                shutil.copy(ticker_folder +file_path, output_file)
            except KeyboardInterrupt:
                break
            except:
                print("An exception occurred with ticker {} ".format(path_ticker))
                faulty_tickers.append(path_ticker)
                continue

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
copy_files_for_dates(filelist_stock_data, output_path, dates_to_check)
