# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 01:16:26 2020

@author: jonsnow
"""


#import matplotlib.pyplot as plt
#import fetch_single_sublist_iqfeed
import std_filter_getter
import sys
#import mplfinance as mpf
#import tensorflow as tf
#from tensorflow import keras
#import numpy as np
#import pandas as pd
#from pandas.plotting import scatter_matrix
#from PIL import Image
#import PIL
#from keras import optimizers
#import tensorflow.keras.backend
#from tensorflow.keras.optimizers import RMSprop
#from keras.preprocessing import image
#from keras.models import model_from_json, load_model
#from keras.layers import Dropout
#from keras.utils import to_categorical
#from tensorflow.keras.models import Sequential, save_model, load_model
from shutil import copyfile
import os
#from sklearn.model_selection import train_test_split
#from datetime import date, datetime, timedelta
import pandas_market_calendars as mcal
#import json
#import datetime
#import time


frame_size = 100
threshold_Y_binary_percent = 1.024
lower_limit = 0.9
upper_limit = 2.9
path_stock_data = "stock_data/stocks_2019_2020"
output_dataframes_folder_initial = "stock_data/filtered_vol_std/"
output_folder = output_dataframes_folder_initial


start_date='2020-09-01'
end_date='2020-10-23'
def get_dates_list_to_check_for_date_range(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    
    dates_to_check = nyse.schedule(start_date=start_date, end_date=end_date)
    dates_to_check = mcal.date_range(dates_to_check, frequency='1D')
    dates_to_check_str = []
    for date_in_list in dates_to_check:
        dates_to_check_str.append(str(date_in_list).split(" ")[0])
    return dates_to_check_str
dates_to_check = get_dates_list_to_check_for_date_range(start_date, end_date)

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

i=0
#current_date = '2020-10-23'
for current_date in dates_to_check:
    results = std_filter_getter.get_result_ticker_std_tuples(current_date, lower_limit, upper_limit, path_stock_data, 0, 0)
    for j in range(len(results)):
        try:
            current_ticker = results[j][0]
            current_source_path = path_stock_data + "/" + current_ticker + "/" + current_date + ".csv"
            current_destination_path = output_dataframes_folder_initial + "l" + str(lower_limit) + "/" + "u" + str(upper_limit) + "/" + current_ticker + "/"
            
            if not os.path.exists(current_destination_path):
                os.mkdir(current_destination_path)
            
            copyfile(current_source_path, current_destination_path + current_date + ".csv")
        except KeyboardInterrupt:
            break
        except:
            print( "Unexpected error:", sys.exc_info()[0])
            print( "Error in ticker ",current_ticker)
    print("filtered date {}".format(current_date))