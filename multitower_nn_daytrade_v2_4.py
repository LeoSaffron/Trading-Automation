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
#import tulipy as ti
import talib


frame_size = 150
frame_size_Y = 0
threshold_Y_binary_percent = 1.04
lower_limit = 0.3
upper_limit = 3.0
min_close_value = 8

min_amplitude_relative = 0.034
max_amplitude_relative = 0.053
min_amplitude_relative = 0.024
max_amplitude_relative = 0.063
max_amplitude_relative = 1.063
#lower_limit = 0.9
#upper_limit = 2.9
folder_limit_for_debug = 0

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
date_split_str = "2020-09-01"



#marketcap_df = pd.read_csv(market_cap_path).set_index('symbol')

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []

def get_amplitude_for_df(df, split_index):
    df_cropped = df[:split_index]
    return float(df_cropped['High'].max() - df_cropped['Low'].min())

def check_df_for_maximum_gain_value(df, split_index, frame_length_Y = 120):
    close_value = df['Close'][split_index-1]
    df_scan = None
    if(frame_length_Y == 0):
        df_scan = df.iloc[split_index-1 :]
    else:
        df_scan = df.iloc[split_index-1 : split_index + frame_length_Y]
    maximum_value = df_scan['High'].max()
    return maximum_value - close_value

def check_df_for_maximum_gain_percent(df, split_index, frame_length_Y = 120):
    close_value = df['Close'][split_index-1]
    df_scan = None
    if(frame_length_Y == 0):
        df_scan = df.iloc[split_index-1 :]
    else:
        df_scan = df.iloc[split_index-1 : split_index + frame_length_Y]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value
    
    
def get_clean_X_Y_percentage_from_df_by_marketcap_volume(df_list, frame_length = 100, frame_length_Y = 120):

    X_list = []
    X_close_list = []
    Y_list = []
    X_amplitude_list = []
    Y_relative_list = []
#    list_to_test = []
#    for i in range(len(df_list)):
#        for j in range(len(df_list[i])):
#            list_to_test.append((i,j))
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            current_df = df_list[i][j]
            current_day_frame_length = len(current_df)
            if(current_day_frame_length < frame_length + frame_length_Y):
                continue
#            current_df = current_df[:frame_length].drop('symbol', axis=1)
            current_df = current_df.drop('symbol', axis=1)
            y_value = check_df_for_maximum_gain_percent(current_df, frame_length, frame_length_Y)
            amplitute = get_amplitude_for_df(current_df, frame_length)
            max_gain_value = check_df_for_maximum_gain_value(current_df, frame_length, frame_length_Y)
            
            current_df = current_df[:frame_length]
            if(y_value == float('inf')):
                continue
            X_list.append(current_df)
            Y_list.append(y_value)
            X_amplitude_list.append(amplitute)
            X_close_list.append(float(current_df['Close'].iloc[frame_length-1]))
            Y_relative_list.append(max_gain_value / amplitute)
            
#            print(current_df)
    return X_list, Y_list, X_close_list, X_amplitude_list, Y_relative_list

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
#folderlist_stock_data = folderlist_stock_data[:2200]
if (folder_limit_for_debug > 0):
    folderlist_stock_data = folderlist_stock_data[:folder_limit_for_debug]
#folderlist_stock_data = [('BCRX', 'stock_data/filtered_vol/l1000000/u6000000/BCRX/')]

def get_X_Y_data_from_folderlist(folder_path, frame_length):
    clean_df = get_stock_data_from_folderlist(folder_path)
    print('getting volume groups df')
    df_X, df_Y, df_X_close, df_X_amplitude, df_Y_relative = get_clean_X_Y_percentage_from_df_by_marketcap_volume(clean_df, frame_length = frame_length, frame_length_Y = frame_size_Y)
    print('fomatting df')
    df_Y_binary = (np.array(df_Y) > threshold_Y_binary_percent)
    return df_X, df_Y, df_X_close, df_X_amplitude, df_Y_binary, df_Y_relative


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
df_all_X_close = []
df_all_X_amplitude = []
df_all_Y_binary = []
df_all_Y_relative = []
test_potential_volatility_list = []

start_process_time = datetime.now()
for i in range(int(len(folderlist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(folderlist_stock_data) / file_batch_size) + 1))
    df_X, df_Y, df_X_close, df_X_amplitude, df_Y_binary, df_Y_relative = get_X_Y_data_from_folderlist(
            folderlist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(folderlist_stock_data))],
            frame_size)
    for x in df_X : 
        df_all_X.append(x) 
    for y in df_Y : 
        df_all_Y.append(y) 
    for x in df_X_close : 
        df_all_X_close.append(x) 
    for x in df_X_amplitude : 
        df_all_X_amplitude.append(x) 
    for y in df_Y_binary : 
        df_all_Y_binary.append(y) 
    for y in df_Y_relative : 
        df_all_Y_relative.append(y) 



end_process_time = datetime.now()
print(end_process_time - start_process_time)


ampitude_relative = []
for i in range(len(df_all_X_amplitude)):
    ampitude_relative.append(df_all_X_amplitude[i] / df_all_X_close[i])
ampitude_relative = np.array(ampitude_relative)

amp_filtered_indices = []
for i in range(len(ampitude_relative)):
    if((ampitude_relative[i] >= min_amplitude_relative) and (ampitude_relative[i] <= max_amplitude_relative)):
        amp_filtered_indices.append(True)
    else:
        amp_filtered_indices.append(False)

date_split_train_test = datetime.strptime(date_split_str, "%Y-%m-%d")
df_dates_list = []
for df in df_all_X:
    df_dates_list.append(datetime.strptime(df.index[0].split(' ')[0], "%Y-%m-%d"))

split_date_threshold_list = []
for df in df_all_X:
    split_date_threshold_list.append(datetime.strptime(df.index[0].split(' ')[0], "%Y-%m-%d") >= date_split_train_test) 
    
def get_train_test_Y_by_threshold(threshold): 
    temp_arr = np.array(df_all_Y) > threshold
    Y_train_temp = []
    Y_test_temp = []
    for i in range(len(df_all_X)):
    #    if (datetime.strptime(df_all_X[i].index[0], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
        if (split_date_threshold_list[i] ==  False):
            Y_train_temp.append(temp_arr[i])
        else:
            Y_test_temp.append(temp_arr[i])
    return Y_train_temp, Y_test_temp

def get_train_test_Y_relative_by_threshold(threshold): 
    temp_arr = np.array(df_all_Y_relative) > threshold
    Y_train_temp = []
    Y_test_temp = []
    indices_arr_temp = []
    for i in range(len(df_all_X)):
        if(amp_filtered_indices[i] == True):
    #    if (datetime.strptime(df_all_X[i].index[0], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
            if (split_date_threshold_list[i] ==  False):
                Y_train_temp.append(temp_arr[i])
            else:
                Y_test_temp.append(temp_arr[i])
                indices_arr_temp.append(i)
    return Y_train_temp, Y_test_temp, indices_arr_temp


def get_train_test_Y_relative_by_threshold_non_binary(threshold): 
    temp_arr = np.array(df_all_Y_relative)
    Y_train_temp = []
    Y_test_temp = []
    for i in range(len(df_all_X)):
        if(amp_filtered_indices[i] == True):
    #    if (datetime.strptime(df_all_X[i].index[0], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
            if (split_date_threshold_list[i] ==  False):
                Y_train_temp.append(temp_arr[i])
            else:
                Y_test_temp.append(temp_arr[i])
    return Y_train_temp, Y_test_temp
#df_all_X, df_all_Y_binary = get_X_Y_data_from_cached_csv(path_cached_df)

#for df in df_all_X:
#    df.to_csv("")
    
#testdf = pd.DataFrame(data=[amp_rel, df_all_X_amplitude, df_all_X_close, df_all_Y_relative, df_all_Y])
#testdf = testdf.T
#testdf.columns = ["amp_relative", "amplitude", "last_close", "y_relative", "y_percent"]
def plot_sub_df_values(df_to_plot, x_column, y_column, indices , limits, bin_number):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    numBins = bin_number
    ax.set_xlim(limits[0], limits[1])
    ax.hist(df_to_plot.sort_values(by=[x_column])[y_column][indices[0]:indices[1]],numBins,color='green')
    plt.show()

start_process_time = datetime.now()
print("adding bollinger bands")
period = 15
multiplier = 2
tempzeros = np.zeros(period - 1)
for i in range(len(df_all_X)):
    df_all_X[i] = np.array(df_all_X[i])
#    bands2 = ti.bbands(df_all_X[i].T[3], period, 2)
#    bands3 = ti.bbands(df_all_X[i].T[3], period, 3)
    bands2 = talib.BBANDS(df_all_X[i].T[3], timeperiod=period, nbdevup=2, nbdevdn=2)
    bands3 = talib.BBANDS(df_all_X[i].T[3], timeperiod=period, nbdevup=3, nbdevdn=3)
    for j in range(len(bands2)):
        bands2[j][:period-1] = tempzeros
    for j in range(len(bands3)):
        bands3[j][:period-1] = tempzeros
    sma = bands2[1]
#    upperBand_Width2 = np.concatenate([tempzeros, bands2[0]])
#    lowerBand_Width2 = np.concatenate([tempzeros, bands2[2]])
#    upperBand_Width3 = np.concatenate([tempzeros, bands3[0]])
#    lowerBand_Width3 = np.concatenate([tempzeros, bands3[2]])
    upperBand_Width2 = bands2[0]
    lowerBand_Width2 = bands2[2]
    upperBand_Width3 = bands3[0]
    lowerBand_Width3 = bands3[2]
    df_all_X[i] = np.array([df_all_X[i].T[0], df_all_X[i].T[1], df_all_X[i].T[2], df_all_X[i].T[3], df_all_X[i].T[4],
                           upperBand_Width2, lowerBand_Width2, upperBand_Width3, lowerBand_Width3,
                           sma]).T
end_process_time = datetime.now()
print(end_process_time - start_process_time)


start_process_time = datetime.now()
print("scaling and rearranging")
#for df_list in df_all_X:
#start_process_time = datetime.now()
for i in range(len(df_all_X)):
#    df_all_X[i] = np.array(df_all_X[i])
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
    
    bollinger_band_width_2_upper[:period-1] = np.array([0] * (period - 1))
    bollinger_band_width_2_lower[:period-1] = np.array([0] * (period - 1))
    bollinger_band_width_3_upper[:period-1] = np.array([0] * (period - 1))
    bollinger_band_width_3_lower[:period-1] = np.array([0] * (period - 1))
    
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


#for i in range(len(df_all_X)):
#    save_cache_path = path_cached_df + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d") + "/df_" + str(i) + ".csv"
#    np.savetxt(save_cache_path, df_all_X[i], delimiter=',')
#np.savetxt(path_cached_df + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d") + ".csv", df_all_Y, delimiter=',')
#np.savetxt(path_cached_df + "/" + datetime.strftime(datetime.now(), "%Y-%m-%d") + "_binary.csv", df_all_Y_binary, delimiter=',')


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
Y_relative_train = []

X_test = []
Y_test = []
Y_relative_test = []

indices_X_test = []
#indices_Y_test = []
date_threshold_train_test = datetime.now() - timedelta(days=90)
for i in range(len(df_all_X)):
#    if (datetime.strptime(df_all_X[i].index[0], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
    if(amp_filtered_indices[i] == True):
        if (split_date_threshold_list[i] ==  False):
            X_train.append(df_all_X[i])
            Y_train.append(df_all_Y_binary[i])
            Y_relative_test.append(df_all_Y_relative[i])
#            indices_X_test.append(i)
        else:
            X_test.append(df_all_X[i])
            Y_test.append(df_all_Y_binary[i])
            Y_relative_test.append(df_all_Y_relative[i])
            indices_X_test.append(i)

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



amp_test = []
for i in range(len(indices_X_test)):
    amp_test.append(df_all_X_amplitude[indices_X_test[i]])


close_test = []
for i in range(len(indices_X_test)):
    close_test.append(df_all_X_close[indices_X_test[i]])

transform_relative_to_percentage = np.array(amp_test) / np.array(close_test)

class MyCustomCallback(tf.keras.callbacks.Callback):
    
    def result_forThreshold(self, predictions, threshold, Y_test):
        p = predictions > threshold
        c = 0
        for i in range(len(p)):
            if(p[i]):
                if(Y_test[i] == True):
                    c += 1
        return c , p.sum()
    
    def get_percentage_predictions_by_relative(self, predictions, Y_test, goal_percentage, goal_relative, nn_threshold):
        result_preds = []
        c = 0
        for i in range(len(predictions)):
            if(goal_relative * transform_relative_to_percentage[i] > goal_percentage ):
                if (predictions[i] > nn_threshold):
                    if(Y_test[i] == True):
                        result_preds.append(True)
                    else:
                        result_preds.append(False)
        result_preds = np.array(result_preds)
        return len(result_preds), result_preds.sum()
    
    def calculate_prediction_results(self, model, predictions, Y_test, verbose = 2):
#        predictions = model.predict(X_test)
#        thresholds = np.linspace(0.5, 0.9, 5)
        thresholds_results = []
        
        total_guesses = []
        for recall_threshold in thresholds:
            total_guesses.append(np.array(predictions > recall_threshold).sum())
        
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            result = self.result_forThreshold(predictions, threshold, Y_test)
            thresholds_results.append(result)
        
        if (verbose == 1):
            for i in range(len(thresholds)):
                correct_guessus = thresholds_results[i][0]
                total_gueses = thresholds_results[i][1]
                print("[th {}. st: {}/{}: {:.2f}% ;] ".format(thresholds[i], correct_guessus, total_gueses,
                      100 * (correct_guessus / total_gueses)), end='')
            print("")
        elif (verbose == 2):
            for i in range(len(thresholds)):
                correct_guessus = thresholds_results[i][0]
                total_gueses = thresholds_results[i][1]
                print("with threshold {} : {} out of {} are correct. percantage: {:.2f}%".format(thresholds[i],
                      correct_guessus, total_gueses, 100 * (correct_guessus / total_gueses)))
        return thresholds_results, total_guesses
    
    
    
    def calculate_prediction_results_percantage(self, model, predictions, Y_test, percent, verbose = 2):
        thresholds_results = []
        total_guesses = []
#        for recall_threshold in thresholds:
#            total_guesses.append(np.array(predictions > recall_threshold).sum())
        
        for i in range(len(thresholds)):
            threshold = thresholds[i]
            result = []
            results_guesses = []
            for relative_goal_index in range(len(goal_relative_list)):
                result_temp = self.get_percentage_predictions_by_relative(predictions,
                                                                          Y_test_relative_list_with_goal_percentage[relative_goal_index],
                                                                          percent, goal_relative_list[relative_goal_index], threshold)
                result.append(result_temp[1] / result_temp[0])
                total_guesses_temp = result_temp[1]
                results_guesses.append(total_guesses_temp)
            thresholds_results.append(result)
            total_guesses.append(results_guesses)
            
        return thresholds_results, total_guesses
                


    def on_epoch_end(self, epoch, logs=None):
#        print(self.model.summary())
#        print("\nlen is {}".format(len(Y_test)))
#        global test_nn, test2
        global epoch_dict, epoch_dict_weights
        nn_stats_array = np.empty(shape=(0,len(thresholds)))
        nn_stats_array = []
#        nn_stats_array_percentage = []
        total_guesses = []
        total_guesses_percentage = []
        predictions = self.model.predict(X_test)
        precision_percent = []
#        predictions_pernectage = predictions * transform_relative_to_percentage
        for i in range(len(goal_relative_list)):
            print("results for {:.2f}%".format( ( goal_relative_list[i]) * 100))
            nn_stats_results  = (self.calculate_prediction_results(
                    self.model, predictions, Y_test_relative_list_with_goal_percentage[i], verbose = 0))
            precision_results = np.array(nn_stats_results[0]).T
            nn_stats_array.append( precision_results[0] / precision_results[1])
            total_guesses = nn_stats_results[1]
        
        for i in range(len(goal_percentage_list)):
            print("results for {:.2f}%".format( ( goal_percentage_list[i]) * 100))
            percent = goal_percentage_list[i]
            nn_stats_results  = (self.calculate_prediction_results_percantage(
                    self.model, predictions, Y_test_relative_list_with_goal_percentage[i], percent, verbose = 0))
#            total_guesses_percentage.append(nn_stats_results[1])
            total_guesses_percentage.append(pd.DataFrame(nn_stats_results[1], columns = goal_relative_list, index = thresholds).T)
            precision_percent.append(pd.DataFrame(nn_stats_results[0], columns = goal_relative_list, index = thresholds).T)
        precision = pd.DataFrame(nn_stats_array, columns=thresholds, index = goal_relative_list)
        print("total guesses:")
        print(total_guesses)
        print("precicios matrix:")
        print(precision)
        for i in range(len(goal_percentage_list)):
            print("results for {:.2f}%\n".format( ( goal_percentage_list[i]) * 100))
            print("total guesses percentage:")
            print(total_guesses_percentage[i])
            print("precicios matrix percentage:")
            print(precision_percent[i])
        epoch_dict.append({'epoch' : epoch,
         'precision_matrix': precision,
         'precision_matrix_percent': precision_percent,
         'total_guesses' : total_guesses,
         'total_guesses_percentage' : total_guesses_percentage
         })
        epoch_dict_weights.append(self.model.get_weights().copy())

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

def get_new_model_seuqntial():
    return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape = (frame_size,frame_width,1)),
            tf.keras.layers.Conv2D(64, (1,2), activation='relu'),
            tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
            tf.keras.layers.Conv2D(16, (2,3), activation='relu'),
            tf.keras.layers.Conv2D(32, (1,2), activation='relu'),
            tf.keras.layers.Conv2D(64, (2,3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.3),
             tf.keras.layers.Dense(10, activation='relu'),
#            tf.keras.layers.Dense(1, activation='sigmoid')
            tf.keras.layers.Dense(1, tf.keras.activations.exponential)
            ])

def get_new_model():
    inputs = tf.keras.layers.Input(shape=(frame_size,frame_width,1))
    x = tf.keras.layers.Conv2D(16, (2,2), activation='relu') (inputs)
    x = tf.keras.layers.Conv2D(64, (1,2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (2,2), activation='relu')(x)
#    x = tf.keras.layers.Conv2D(16, (2,3), activation='relu')(x)
#    x = tf.keras.layers.Conv2D(32, (1,2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (2,3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(20, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
#    tf.keras.layers.Dense(1, activation='sigmoid')(inputs)
    x = tf.keras.layers.Dense(1, tf.keras.activations.exponential)(x)
    return tf.keras.Model(inputs, x)

    

#model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy',  metrics=['accuracy', precision, recall])

#model_binary.compile( optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8),
#                     loss='categorical_crossentropy',  metrics=['accuracy', precision, recall])
#model_binary.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                     loss='mse',  metrics=['accuracy', precision, recall])




        

model_binary2 = get_new_model()
thresholds = np.linspace(0.5, 0.9, 5)
thresholds = np.linspace(0.1, 0.9, 9)
#thresholds = np.linspace(0.01, 0.09, 9)
goal_percentage_list = [1.03, 1.024, 1.02, 1.015, 1.01]
goal_percentage_list = [0.015, 0.02, 0.025, 0.03]
goal_relative_list = [0.6, 0.8, 1.0, 1.5, 2, 2.5, 3, 4]
goal_relative_list = [0.1, 0.3, 0.6, 0.8, 1.0]
goal_relative_list = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 1.5, 2, 2.5, 3, 4]
#goal_percentage_list = [0.3, 0.9, 1.5]
Y_test_list_with_goal_percentage = []
Y_test_relative_list_with_goal_percentage = []
#for item in goal_percentage_list:
#    Y_test_list_with_goal_percentage.append(get_train_test_Y_by_threshold(item)[1])
#for item in goal_percentage_list:
for item in goal_relative_list:
    Y_test_relative_list_with_goal_percentage.append(get_train_test_Y_relative_by_threshold(item)[1])
#thresholds = np.linspace(0.01, 0.09, 9)
Y_train, Y_test = get_train_test_Y_by_threshold(1.024)
Y_train, Y_test, test_indices = get_train_test_Y_relative_by_threshold(0.24)
#Y_train, Y_test = get_train_test_Y_relative_by_threshold_non_binary(0.40)
Y_train_2_4, Y_test_2_4 = get_train_test_Y_by_threshold(1.014)

epoch_dict = []
epoch_dict_weights = []
learning_rate = 0.0002
epochs = 10000
Y_multiplier = 0.1
model_binary2.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='mse',  metrics=['accuracy', precision, recall])
#model_binary.compile( optimizer="adam", loss=precision,  metrics=['accuracy', precision, recall])
#history_binary = model_binary.fit(np.array(X_train),np.array(Y_train),verbose=1,batch_size=32,  epochs=50,validation_data=(X_test,Y_test))
#history_binary = model_binary.fit(X_train,to_categorical(np.array(Y_train)),verbose=1,batch_size=32, shuffle=True,
#                                  epochs=30,validation_data=(X_test,to_categorical(np.array(Y_test))))
import math
Y_feed = np.array(Y_train) 
#Y_feed = Y_train + (np.array(Y_train) - 0.5) * Y_multiplier
#Y_feed = np.array(Y_train) * Y_multiplier

#Y_feed = (np.array(Y_train) - 0.5) * Y_multiplier
#Y_feed = []
#for x in Y_train:
##    Y_feed.append(1.01 ** (0.5 + x) - 0.5)
#    Y_feed.append(1.31 ** (x - 0.5) )
##    Y_feed.append(1.01 **  x)
#Y_feed = np.array(Y_feed)
history_binary2 = model_binary2.fit(X_train, Y_feed,verbose=1,
                                    steps_per_epoch=600,batch_size=64, shuffle=True,
                                  epochs=epochs,validation_data=(X_test,np.array(Y_test)),
                                  callbacks=[MyCustomCallback()])
#df_all_Y_relative
#history_binary2 = model_binary2.fit(X_train, Y_multiplier * np.array(Y_train),verbose=1,
#                                    batch_size=64, shuffle=True,
#                                  epochs=epochs,validation_data=(X_test,np.array(Y_test)),
#                                  callbacks=[MyCustomCallback()])

#history_binary = model_binary2.fit(X_train,np.array(Y_train),verbose=1,batch_size=64, shuffle=True,
#                                  epochs=50,validation_data=(X_test,np.array(Y_test)))

#test_date_list = pd.to_datetime(pd.read_csv("examples/fetched/ZM/2020_04_16.csv").date[:25])

t = []
#for x in epoch_dict[1286:1290]:
for x in epoch_dict[0:200]:
    t.append(100 * x['precision_matrix_percent'][3][0.5])
plt.plot(t)
t2 = []
for x in epoch_dict[00:200]:
    t2.append(x['total_guesses_percentage'][2][0.4])
plt.plot(t2)


precision_matrix

plt.plot(t)
plt.plot(t2)

t3 = []
for x in range(len(t)):
    t3.append(t[x] + t2[x])

plt.plot(t3)



t = []
#for x in epoch_dict[1286:1290]:
for x in epoch_dict[0:200]:
    t.append(100 * x['precision_matrix'][0.9])
plt.plot(t)


t = []
#for x in epoch_dict[1286:1290]:
for x in epoch_dict[130:140]:
    t.append(100 * x['precision_matrix_percent'][3][0.4][0.2])
plt.plot(t)
t2 = []
for x in epoch_dict[130:140]:
    t2.append(x['total_guesses_percentage'][3][0.4][0.2])
plt.plot(t2)

t3 = []
for x in range(len(t)):
    if((t[x] > 60) and (t2[x] > 60)):
        t3.append(t[x] + t2[x])
    else:
        t3.append(0)
plt.plot(t3)

model3 = get_new_model()
model3.set_weights(epoch_dict_weights[132])
ppp = model3.predict(X_test)
predssss = ppp > 0.4
c = 0
for oi in range(len(predssss)):
    o = predssss[oi]
    if(o == True):
        if(0.2 * transform_relative_to_percentage[oi] >= 0.03):
            c += 1
            #print(oi)
print(c)

model3.set_weights(epoch_dict_weights[132])
count3 = []
for i in range(50,100):
    model3.set_weights(epoch_dict_weights[i])
    ppp = model3.predict(X_test)
    predssss = ppp > 0.5
    c = 0
    for oi in range(len(predssss)):
        o = predssss[oi]
        if(o == True):
            if(0.5 * transform_relative_to_percentage[oi] >= 0.03):
                c += 1
                #print(oi)
    count3.append(c)
    print(c)







def result_forThreshold(predictions, Y_test, threshold):
    p = predictions > threshold
    
    
    c = 0
    for i in range(len(p)):
        if(p[i]):
            if(Y_test[i] == True):
                c += 1
    return c , p.sum()
    
def print_results_for_model(model, X_test, goal_percentage):
    predictions = model.predict(X_test)
    Y_test = get_train_test_Y_by_threshold(goal_percentage)[1]
#    threshold = 0.5
#    p = predictions > threshold
#    
#    
#    c = 0
#    for i in range(len(p)):
#        if(p[i]):
#            if(Y_test[i] == True):
#                c += 1
    
#    print("with threshold {} : {} out of {} are correct".format(threshold, c, p.sum()))
    print("results for goal {}%".format(goal_percentage))
    

        
    thresholds = np.linspace(0.4, 0.9, 6)
    thresholds_results = []
    
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        result = result_forThreshold(predictions, Y_test, threshold)
        thresholds_results.append(result)
    
#    result_05 = result_forThreshold(predictions, Y_test, 0.5)
    
    for i in range(len(thresholds)):
        correct_guessus = thresholds_results[i][0]
        total_gueses = thresholds_results[i][1]
        print("with threshold {} : {} out of {} are correct. percantage: {:.2f}%".format(thresholds[i],
              correct_guessus, total_gueses, 100 * correct_guessus / total_gueses))


print_results_for_model(model_binary2, X_test, 1.03)
print_results_for_model(model_binary2, X_test, 1.024)
print_results_for_model(model_binary2, X_test, 1.02)
print_results_for_model(model_binary2, X_test, 1.015)
print_results_for_model(model_binary2, X_test, 1.01)

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
#thresholds = thresholds
#thresholds_results = thresholds_results
last_val_precision = history_binary2.history['val_precision'][-1:][0]
last_val_recall = history_binary2.history['val_recall'][-1:][0]
#last_percentage = 100 * result_05[0] / result_05[1]

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
#        'thresholds' : str(thresholds),
#        'thresholds_results' : str(thresholds_results),
        'last_val_precision' : str(last_val_precision),
        'last_val_recall' : str(last_val_recall)
        }





test_nn = []
test2 = []






#current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
#output_save_model_folder = models_folder + "/" + current_time + "/"
#if not os.path.exists(output_save_model_folder):
#    os.mkdir(output_save_model_folder)
#
#model_binary2.save(output_save_model_folder + 'model.h5')
#model_binary2.save_weights(output_save_model_folder + 'weights.h5')
#
#import json
#json_results = json.dumps(results_dict)
#f = open(output_save_model_folder + 'results.json',"w")
#f.write(json_results)
#f.close()




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
