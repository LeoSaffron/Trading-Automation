# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 01:56:50 2020

@author: jonsnow
"""

# importing module  
import csv 
#
#import matplotlib.pyplot as plt
import sys
#import mplfinance as mpf
#import numpy as np
#import pandas as pd
#
#import requests
##from requests.auth import HTTPDigestAuth
#import json
#from datetime import datetime
#
##%matplotlib inline
##import pandas_datareader
#import datetime
#from pandas.plotting import scatter_matrix
#import time
##import pandas as pd
##from sklearn.preprocessing import MinMaxScaler
#from PIL import Image
#import PIL
#import tensorflow as tf
#
#from keras import optimizers
#import tensorflow.keras.backend
#from tensorflow.keras.optimizers import RMSprop
#from keras.preprocessing import image
##from yahoofinancials import YahooFinancials
#
#import json
#from keras.models import model_from_json, load_model
#from datetime import date
#import os
#from sklearn.model_selection import train_test_split
#from keras.layers import Dropout
from datetime import datetime, timedelta
import os
   
# csv fileused id Geeks.csv 
#frame_size = 40
threshold_Y_binary_percent = 1.020
lower_limit = 1000000
upper_limit = 6000000


#path_tickers_list = "./middle_volatility_potential_top100.csv"
path_stock_data = "stock_data/filtered_vol/l" + str(lower_limit) + '/u' + str(upper_limit)
output_path_stock_data = "stock_data/filtered_vol_with_full_rows/l" + str(lower_limit) + '/u' + str(upper_limit)
#market_cap_path = "tickers_with_market_cap.csv"

#filename=path_stock_data + "/AAL/2020-03-09.csv"
def get_initial_input_folderlist(input_dir):
    folderlist = []
    for item in get_immediate_subdirectories(input_dir):
        folderlist.append(path_stock_data + '/' + item)
    return folderlist

def get_initial_input_folderlist_inside(input_dir):
    folderlist = []
    for item in get_immediate_subdirectories(input_dir):
        folderlist.append( item)
    return folderlist

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def fill_single_day(input_file_path, output_file_path):
    last_date = 0
    df = []
    line_number = 0
    one_min_delta = timedelta(minutes=1)
#    fixed_rows = 0
    with open(input_file_path,'r') as data:
        for line in csv.reader(data):
            if(line_number == 1):
                last_date = datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
            if(line_number > 1):
                date = datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
                if(date - last_date != one_min_delta):
#                    new_line_template = df[line_number -1 + fiex_rows].copy()
                    new_line_template = df[-1].copy()
#                    print(line)
                    last_close_value = new_line_template[4]
#                    print(last_close_value)
                    new_line_template[1] = last_close_value
                    new_line_template[2] = last_close_value
                    new_line_template[3] = last_close_value
                    new_line_template[5] = 0
                    row_count_to_fix = int((date - last_date) / one_min_delta)
                    for i in range(1, row_count_to_fix):
                        new_line = new_line_template.copy()
                        new_line[0] = (last_date + one_min_delta * i)
#                        print(new_line[0])
                        df.append(new_line)
#                    fixed_rows += row_count_to_fix 
                    
                last_date = date
            line_number += 1
            df.append(line)
    file = open(output_file_path, 'w+', newline ='') 
    with file:     
        write = csv.writer(file) 
        write.writerows(df) 

folderlist = get_initial_input_folderlist(path_stock_data)
folderlist_tickers = get_initial_input_folderlist_inside(path_stock_data)
for item in folderlist_tickers:
    current_folder = output_path_stock_data + "/" + item
    if not os.path.exists(current_folder):
        os.mkdir(current_folder)
files = []
count_folders = 0
for item in folderlist:
    filelist  = [name for name in os.listdir(item)]
    for f in filelist:
        output_path = output_path_stock_data + (item + "/" + f).split(path_stock_data)[1]
#        files.append(item + "/" + f)
        fill_single_day(item + "/" + f, output_path)
    count_folders += 1
    if(count_folders % 40 == 0):
        print("finished {}".format(count_folders))
    
#output_filename="test_folder" + "/AAL-2020-03-09.csv"
#fill_single_day("stock_data/filtered_vol/l" + str(lower_limit) + '/u' + str(upper_limit) + "/AAL/2020-03-12.csv", "test_folder" + "/AAL-2020-03-12.csv")
#print(df)
#            print(line) 
      
# then data is read line by line  
# using csv.reader the printed  
# result will be in a list format  
# which is easy to interpret