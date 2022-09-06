# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:06:06 2020

@author: jonsnow
"""

#import matplotlib.pyplot as plt
import sys
#import mplfinance as mpf
#import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
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
import os
#from sklearn.model_selection import train_test_split
from datetime import date, datetime, timedelta
#import json
import datetime
#import time


frame_size = 100
threshold_Y_binary_percent = 1.024
lower_limit = 0.01
upper_limit = 0.1

#path_prefix = "C:/frais/datasets/stocks_cache/"
path_prefix = ''
path_tickers_list = "./middle_volatility_potential_top100.csv"
#path_stock_data = "stock_data/filtered_vol/l" + str(lower_limit) + '/u' + str(upper_limit)
#path_stock_data = "stock_data/filtered_vol_std/l" + str(lower_limit) + '/u' + str(upper_limit)
#market_cap_path = "tickers_with_market_cap.csv"
path_cached_df = "C:/frais/datasets/stocks_cache/cached_df/201008_df_before_split"
models_folder = "measured_stats/models_results/trivial_cnn"
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_10_09__02_42_53"
path_vix_daily = "stock_data/vixcurrent_daily.csv"
path_vix_daily = "stock_data/vix2.csv"
#path_last_month_folder = "stock_data/september_part_data/"
path_last_month_folder = "stock_data/filtered_vol_std/"
#filtered_vol_std
#path_2nd_compound_model = "test_model_nn_v1.h5"
#path_2nd_compound_model = "test_2nd_model_2020_10_14.h5"
path_2nd_compound_model = "test_2nd_model_2020_10_14_v2.h5"
path_stock_data = path_last_month_folder + "l" + str(lower_limit) + '/u' + str(upper_limit)
path_stock_data = path_prefix + path_stock_data
vix_daily_min = 8.56
vix_daily_mean = 10.323445
colse_divided_by_open_min = 0.5
close_divided_by_subframe_min = 0.8
close_divided_by_subframe_factor = 2.5
minimal_price_value = 8

#marketcap_df = pd.read_csv(market_cap_path).set_index('symbol')

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

faulty_tickers = []

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value

def check_df_for_end_of_day_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    last_value = df['Open'][-1:][0]
    return (last_value - close_value) / close_value
    
    
def get_clean_X_Y_percentage_from_df_by_marketcap_volume(df_list, frame_length = 100):

    X_list = []
    Y_list = []
    list_to_test = []
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            list_to_test.append((i,j))
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            current_df = df_list[i][j]
            current_day_frame_length = len(current_df)
            if(current_day_frame_length < frame_length):
                continue
#            current_df = current_df[:frame_length].drop('symbol', axis=1)
            current_df = current_df.drop('symbol', axis=1)
            y_value = check_df_for_maximum_gain_percent(current_df, frame_length)
            current_df = current_df[:frame_length]
            if(y_value == float('inf')):
                continue
            X_list.append(current_df)
            Y_list.append(y_value)
            
#            print(current_df)
    return X_list, Y_list        

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
            print("An exception occurred with ticker {} function get_stock_data_from_folderlist".format(path_ticker))
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
    
def get_vix_fdaily_from_file_and_scale(path, min_value, mean):
    vix_daily = pd.read_csv(path, index_col='Date')
    vix_daily = vix_daily - min_value
    vix_daily = vix_daily / (2 * mean)
    return vix_daily

def convert_date_from_df_format_to_vix_format(date_to_convert):
    date_original = datetime.datetime.strptime(date_to_convert.split(' ')[0], "%Y-%m-%d")
    new_year = str(date_original.year)
    new_month = str(int(date_original.month))
    new_day = str(int(date_original.day))
    return new_month + '/' + new_day + '/' + new_year

def get_vix_in_df_by_index(index_in_df, df_list, vix_df):
    date_in_vix = convert_date_from_df_format_to_vix_format(df_list[index_in_df].index[0])
    return vix_df.loc[date_in_vix][0]    

def get_vix_for_single_df(df, vix_df):
    date_in_vix = convert_date_from_df_format_to_vix_format(df.index[0])
    return vix_df.loc[date_in_vix][0] 

def get_close_divided_by_max_in_single_df(df):
    min_value = df['Low'].min()
    max_value = df['High'].max() - min_value
    close_value = df['Close'][-1:][0] - min_value
    return close_value / max_value

def get_close_divided_by_max_in_df_list(df_list):
    output_list = []
    for df in df_list:
        output_list.append(get_close_divided_by_max_in_single_df(df))
    return output_list

def get_close_divided_by_open_in_single_df(df):
    open_value = df['Open'].max()
    close_value = df['Close'][-1:][0]
    return close_value / open_value

def get_close_divided_by_open_in_df_list(df_list, colse_divided_by_open_min):
    output_list = []
#    i = 0
    for df in df_list:
#        print(i)
        output_list.append(get_close_divided_by_open_in_single_df(df) - colse_divided_by_open_min)
#        i+=1
    return output_list

def get_close_divided_by_open_subframe_in_single_df(df, subframe_length = 30):
    open_subframe_value = df['Close'][-31:-subframe_length][0]
    close_value = df['Close'][-1:][0]
    return close_value / open_subframe_value


def get_close_divided_by_open_subframe_in_df_list(df_list, close_divided_by_subframe_min, close_divided_by_subframe_factor, subframe_length = 30):
    output_list = []
    for df in df_list:
        output_list.append(close_divided_by_subframe_factor *
                           (get_close_divided_by_open_subframe_in_single_df(df, subframe_length) - close_divided_by_subframe_min))
    return output_list


#folderlist_stock_data = [('BCRX', 'stock_data/filtered_vol/l1000000/u6000000/BCRX/')]

def filter_df_list_by_days_passed(df_list, days_ago):
    date_threshold_train_test = datetime.datetime.now() - timedelta(days=days_ago)
    filtered_df_list_by_day = []
    for i in range(len(df_list)):
        filtered_by_day_df_for_a_ticker = []
        for j in range(len(df_list[i])):
            if (datetime.datetime.strptime(df_list[i][j].index[0], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
                pass
            else:
                filtered_by_day_df_for_a_ticker.append(df_list[i][j])
        if (len(filtered_by_day_df_for_a_ticker) > 0):
            filtered_df_list_by_day.append(filtered_by_day_df_for_a_ticker)
    return filtered_df_list_by_day

def get_X_Y_data_from_folderlist(folder_path, frame_length):
    clean_df = get_stock_data_from_folderlist(folder_path)
    print('getting volume groups df')
    df_X = filter_df_list_by_days_passed(clean_df, 40)
    return df_X


def get_X_Y_data_from_cached_csv(folder_path, size=0):
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
                if(size != 0 and count_read_files >= size):
                    break
                if (count_read_files % 5000 == 0):
                    print("read {} x dataframes".format(count_read_files))
                i += 1
        except KeyboardInterrupt:
            break
            pass
        except:
            print("An exception occurred")
    list_y_values = list(pd.read_csv(folder_path + ".csv")['0'])
    if(size > 0):
        list_y_values = list_y_values[:size]
    return dflist, list_y_values

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
        frame_date = df.index[0].split(' ')[0]
        print(frame_date)
#        frame_date = str(df.index[0])
        try:
            outdir_temp = output_dataframes_folder  + str(ticker) + '/'
            print("outdir_temp", outdir_temp)
            if not os.path.exists(outdir_temp):
                os.mkdir(outdir_temp)
            output_file_path = outdir_temp + frame_date + ".csv"
            print("output_file_path", output_file_path)
            df.to_csv(output_file_path)
        except:
            print( "Unexpected error:", sys.exc_info()[0])
            print(outdir_temp)
            print( "Error in ticker {}",ticker)

file_batch_size = 40
df_rawframe_X = []
df_all_Y_binary = []
test_potential_volatility_list = []

folderlist_stock_data = get_stock_data_folderlist_from_folder(path_stock_data)

for i in range(int(len(folderlist_stock_data) / file_batch_size) + 1):
    print("loading batch {} out of {}".format(i + 1, int(len(folderlist_stock_data) / file_batch_size) + 1))
    df_X = get_X_Y_data_from_folderlist(
            folderlist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(folderlist_stock_data))],
            frame_size)
    for x in df_X : 
#        print()
        if(x[0]['Close'][0] >= minimal_price_value):
            df_rawframe_X.append(x) 

date_list = []
for dflist in df_rawframe_X:
    for df in dflist:
        date_list.append(df.index[0])
        
#df_cached_trivial_frame_X, df_all_Y_binary = get_X_Y_data_from_cached_csv(path_cached_df, size=len(df_rawframe_X))
model_trivial_dnn = keras.models.load_model(
    path_trivial_cnn +"/model.h5",
    custom_objects=None,
    compile=False
)
subsequent_model = keras.models.load_model(
     path_2nd_compound_model,
    custom_objects=None,
    compile=False
)
#df_rawframe_X
#trivial_nn_score = model_trivial_dnn.predict(np.array(df_cached_trivial_frame_X).reshape(len(df_cached_trivial_frame_X), frame_size, 10, 1))
#trivial_nn_score =  trivial_nn_score.reshape((len(trivial_nn_score)))
#std_of_same_day_list = []
#for df in df_rawframe_X:
#    std_of_same_day_list.append(df.drop('Volume', axis =1).std().mean() * 3)
vix_daily = get_vix_fdaily_from_file_and_scale(path_vix_daily, vix_daily_min, vix_daily_mean)
#vix_in_df_list = []
#for i in range(len(df_rawframe_X)):
#    vix_in_df_list.append(get_vix_in_df_by_index(i, df_rawframe_X, vix_daily))
#
#print(1)
#close_divided_by_max_list = get_close_divided_by_max_in_df_list(df_rawframe_X)
#print(2) 
#close_divided_by_open_list = get_close_divided_by_open_in_df_list(df_rawframe_X, colse_divided_by_open_min) 
#print(3)
#close_divided_by_subframe_open_list = get_close_divided_by_open_subframe_in_df_list(df_rawframe_X,
#                                                                                    close_divided_by_subframe_min, close_divided_by_subframe_factor,  30)    
#print(4)
#
#
#df_all_X = pd.DataFrame([date_list, trivial_nn_score, std_of_same_day_list, vix_in_df_list,
#                         close_divided_by_max_list, close_divided_by_open_list, close_divided_by_subframe_open_list]).T
#df_all_X.columns = ['date', 'first_nn_score', 'volatility', 'vix', 'close_percent_of_max',
#                    'close_percent_of_open', 'close_percent_of_open_subframe']
#df_all_X = df_all_X.set_index('date')





#
#for dflist in df_rawframe_X:
#    save_files(dflist, path_last_month_folder, lower_limit, upper_limit)
#path_last_month_folder



#
###########################
########### SCALE
###########################
#
#
#print("scaling")
##for df_list in df_all_X:
def scale_df_for_trivial_cnn_dataframe(df_to_scale):
    df = df_to_scale.copy()
    high = df.drop(['Volume'], axis=1)['High'].max()
    low = df.drop(['Volume'], axis=1)['Low'].min()
    delta = high - low
    df['Open'] =  (df['Open'] - low) / delta
    df['High'] =  (df['High'] - low) / delta
    df['Low'] =  (df['Low'] - low) / delta
    df['Close'] =  (df['Close'] - low) / delta
    
    high_volume = df['Volume'].max()
    low_volume = df['Volume'].min()
    delta_volume = high_volume - low_volume
    df['Volume'] =  (df['Volume'] - low_volume) / delta_volume
    return df



###########################
######### RERRANGE
###########################

def duplicate_columns_in_df_for_trivial_cnn_dataframe(df_to_scale):
    df = df_to_scale.copy()
    df['High2'] = df['High'].copy()
    df['Close2'] = df['Close'].copy()
    df['Open2'] = df['Open'].copy()
    df['Low2'] = df['Low'].copy()
    return df

def switch_column_positions_in_df_for_trivial_cnn_dataframe(df_to_scale):
    df = df_to_scale.copy()
    df = df.reindex(columns= ['High', 'High2', 'Close', 'Close2', 'Open', 'Open2', 'Low', 'Low2'])
    return df

def add_padding_in_df_for_trivial_cnn_dataframe(df_to_scale):
    df = df_to_scale.copy()
    df.insert(loc=0, column='padding 0', value=np.zeros(df.shape[0]))
    df.insert(loc=9, column='padding 9', value=np.zeros(df.shape[0]))
    return df

def trasfroem_df_to_feed_to_trivial_cnn(df_to_transform, frame_size):
    transformed_df = scale_df_for_trivial_cnn_dataframe(df_to_transform[:frame_size])
    transformed_df = duplicate_columns_in_df_for_trivial_cnn_dataframe(transformed_df)
    transformed_df = switch_column_positions_in_df_for_trivial_cnn_dataframe(transformed_df)
    transformed_df = add_padding_in_df_for_trivial_cnn_dataframe(transformed_df) 
    transformed_df = np.array(transformed_df)
    transformed_df = transformed_df.reshape((1, frame_size, 10, 1))
    return transformed_df


def get_single_df_to_feen_to_subsequent_nn(df):
    df_cropped = df[:frame_size].copy()
    transformed_df = trasfroem_df_to_feed_to_trivial_cnn(df, frame_size)
#    X_train = X_train.reshape(X_train.shape[0], parameter_length, 1)
    prediction_trivial_cnn = model_trivial_dnn.predict(transformed_df)[0][0]
    current_df_date = df_cropped.index[0]
    current_df_volatility = df_cropped.drop('Volume', axis =1).std().mean() * 3
    current_df_vix = get_vix_for_single_df(df_cropped, vix_daily)
    current_df_close_percent_of_max = get_close_divided_by_max_in_single_df(df_cropped)
    current_df_close_percent_of_open = get_close_divided_by_open_in_single_df(df_cropped)
    current_df_close_percent_of_open_subframe = get_close_divided_by_open_subframe_in_single_df(df_cropped, subframe_length = 30)
    df_to_feed = pd.DataFrame([current_df_date, prediction_trivial_cnn, current_df_volatility, current_df_vix,
                         current_df_close_percent_of_max, current_df_close_percent_of_open, current_df_close_percent_of_open_subframe]).T
    df_to_feed.columns = ['date', 'first_nn_score', 'volatility', 'vix', 'close_percent_of_max',
                    'close_percent_of_open', 'close_percent_of_open_subframe']
    df_to_feed = df_to_feed.set_index('date')
    return df_to_feed

from enum import Enum
class Target_or_stop_loss_return_value(Enum):
    STAYED_WITHIN_LIMITS = 0
    TARGET_REACHED = 1
    STOP_LOSS_REACHED = 2

def check_df_for_target_or_stoploss(df, target_goal_percent, stop_loss_percent, split_index):
    target_goal_percent += 1
    stop_loss_percent = 1 - stop_loss_percent
    
    close = df.iloc[split_index - 1]['Close']
    df_post_division = df[split_index:]
    target_gain_price = close * target_goal_percent
    stop_loss_price = stop_loss_percent * close
    for i in range(len(df_post_division)):
        line = df_post_division.iloc[i]
        
        if(max(line['Close'], line['Open']) > target_gain_price):
            return Target_or_stop_loss_return_value.TARGET_REACHED, i + split_index
#            break
        if(min(line['Close'], line['Open']) <= stop_loss_price):
            return Target_or_stop_loss_return_value.STOP_LOSS_REACHED, i + split_index
    return Target_or_stop_loss_return_value.STAYED_WITHIN_LIMITS, len(df_post_division) - 1

#threshold = 0.6
#count_success = 0
#count_failure = 0
#successes_value = []
#failures_value = []
#percent_won = 0
#percent_lost = 0
#
#for i in range(len(df_rawframe_X)):
#    for df in df_rawframe_X[i]:
#        if(len (df) > frame_size):
#            total_counts = count_success + count_failure
#                
#            df_to_feed = get_single_df_to_feen_to_subsequent_nn(df)
#            parameter_length = len(df_to_feed.columns)
#            array_to_feed_subsequent_nn = np.array(df_to_feed, dtype=float).reshape(1, parameter_length, 1)
#            nn_score = subsequent_model.predict(array_to_feed_subsequent_nn)
#            if (nn_score > threshold):
#                if(total_counts % 20 == 0 and total_counts != 0):
#                    print("evalutated {} frames".format(total_counts))
#                    print("seccess-fail {}-{}. gains: {:.3f}% losses: {:.3f}%".format(count_success, count_failure, percent_won, percent_lost))
#                    
#                max_gain = check_df_for_maximum_gain_percent(df, frame_size)
#                if(max_gain > threshold_Y_binary_percent):
#                    count_success += 1
#                    successes_value.append(threshold_Y_binary_percent - 1)
#                    percent_won += (threshold_Y_binary_percent - 1)
#                else:
#                    count_failure += 1
#                    failures_value.append(-1 * check_df_for_end_of_day_gain_percent(df, frame_size))
#                    percent_lost -= (check_df_for_end_of_day_gain_percent(df, frame_size))
#    
#100 * (percent_won - percent_lost) / (count_success + count_failure)

def evaluate_dl_list_for_reslusts(df_list, nn_threshold, target_goal_percent,
                                  stop_loss_percent, verbose = 1):
    threshold = 0.6
    target_goal_percent = 0.02
    stop_loss_percent = 0.01
    count_success = 0
    count_failure = 0
    successes_value = []
    failures_value = []
    percent_won = 0
    percent_lost = 0
    verbose = 1
    
    for i in range(len(df_rawframe_X)):
        for df in df_rawframe_X[i]:
            if(len (df) > frame_size):
                total_counts = count_success + count_failure
                    
                df_to_feed = get_single_df_to_feen_to_subsequent_nn(df)
                parameter_length = len(df_to_feed.columns)
                array_to_feed_subsequent_nn = np.array(df_to_feed, dtype=float).reshape(1, parameter_length, 1)
                nn_score = subsequent_model.predict(array_to_feed_subsequent_nn)
                if (nn_score > threshold):
                    if (verbose == 1):
                        if(total_counts % 20 == 0 and total_counts != 0):
                            print("evalutated {} frames".format(total_counts))
                            print("seccess-fail {}-{}. gains: {:.3f}% losses: {:.3f}%".format(count_success, count_failure,
                                  100 * percent_won, 100 * percent_lost))
                    
                    current_division_close = df.iloc[frame_size - 1]['Close']
                    goals_result, index_result = check_df_for_target_or_stoploss(df, target_goal_percent, stop_loss_percent, frame_size)
                    sell_price = df.iloc[index_result]['Close']
                    result_in_price = sell_price - current_division_close
                    result_in_percent = result_in_price / current_division_close
                    current_df_win = False
                    if(goals_result == Target_or_stop_loss_return_value.TARGET_REACHED):
                        current_df_win = True
                    elif(goals_result == Target_or_stop_loss_return_value.STAYED_WITHIN_LIMITS):
                        if(df.iloc[-1:]['Close'][0] > current_division_close):
                            current_df_win = True
    #                max_gain = check_df_for_maximum_gain_percent(df, frame_size)
                    if(current_df_win):
                        count_success += 1
                        successes_value.append(result_in_percent)
                        percent_won += (result_in_percent)
                    else:
                        count_failure += 1
                        failures_value.append(-1 * result_in_percent)
                        percent_lost -= (result_in_percent)
    return count_success, count_failure, successes_value, failures_value, percent_won, percent_lost
    
#100 * (percent_won - percent_lost) / (count_success + count_failure)




def get_indices_in_df_for_specific_date(df_list, date):
    filtered_df_list = []
    for i in range(len(df_list)):
        for j in range(len(df_list[i])):
            if(check_if_date_in_df_fits(df_list[i][j], date)):
                filtered_df_list.append((i,j))
    return filtered_df_list

def check_if_date_in_df_fits(df, date):
#    try:
        date_df = df.index[0].split(" ")[0]
        return date == date_df

import pandas_market_calendars as mcal

# Create a calendar
start_date='2020-09-24'
end_date='2020-10-22'

def get_dates_list_to_check_for_date_range(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    
    dates_to_check = nyse.schedule(start_date=start_date, end_date=end_date)
    dates_to_check = mcal.date_range(dates_to_check, frequency='1D')
    dates_to_check_str = []
    for date_in_list in dates_to_check:
        dates_to_check_str.append(str(date_in_list).split(" ")[0])
    return dates_to_check_str
dates_to_check = get_dates_list_to_check_for_date_range(start_date, end_date)

def get_indices_to_evaluate_by_date(df_list, date, limit_per_day, threshold = 0.5, delay_minutes = 0, minutes_before_eod_to_sell = 0):
    current_day_indices_with_nn_score = []
    for i,j in get_indices_in_df_for_specific_date(df_list, date):
        
        df = df_list[i][j]
#        print(" i {} j {}".format(i,j))
        if(len (df) <= frame_size + delay_minutes + minutes_before_eod_to_sell):
            continue
        
        df_to_feed = get_single_df_to_feen_to_subsequent_nn(df)
        parameter_length = len(df_to_feed.columns)
        array_to_feed_subsequent_nn = np.array(df_to_feed, dtype=float).reshape(1, parameter_length, 1)
        nn_score = subsequent_model.predict(array_to_feed_subsequent_nn)
#        print(nn_score)
        if (nn_score > threshold):
#            print(" i {} j {} score {}".format(i,j, nn_score[0][0]))
            current_day_indices_with_nn_score.append((i, j, nn_score[0][0]))
        del(df)
    result_list = []
    current_day_indices_with_nn_score = pd.DataFrame(current_day_indices_with_nn_score)
    if(len(current_day_indices_with_nn_score) > 0):
        current_day_indices_with_nn_score.columns=['i', 'j', 'score']
        sorted_df = current_day_indices_with_nn_score.sort_values(by='score')
        if (limit_per_day > 0):
            sorted_df = sorted_df[-limit_per_day:]
        for i in range(len(sorted_df)):
            current_line = sorted_df.iloc[i]
            result_list.append((int(current_line.iloc[0]), int(current_line.iloc[1]), current_line.iloc[2]))
    return result_list

def evaluate_cached_dl_list_for_reslusts(cached_indices_list, nn_threshold, target_goal_percent,
                                  stop_loss_percent, verbose = 1):
    count_success = 0
    count_failure = 0
    successes_value = []
    failures_value = []
    percent_won = 0
    percent_lost = 0
    for index_in_cached_list in range(len(cached_indices_list)):
        for i,j, score in cached_indices_list[index_in_cached_list]:
            df = df_rawframe_X[i][j]
            nn_score = score
            total_counts = count_success + count_failure
            if (nn_score > threshold):
                if (verbose >= 2):
                    if(total_counts % max_trades_per_day == 0 and total_counts != 0):
                        print(str(datetime.datetime.now()).split('.')[0])
                        print("evalutated {} frames".format(total_counts))
                        print("seccess-fail {}-{}. gains: {:.3f}% losses: {:.3f}%".format(count_success, count_failure,
                              100 * percent_won, 100 * percent_lost))
                
                current_division_close = df.iloc[frame_size - 1]['Close']
                goals_result, index_result = check_df_for_target_or_stoploss(df, target_goal_percent, stop_loss_percent, frame_size)
                sell_price = df.iloc[index_result]['Close']
                result_in_price = sell_price - current_division_close
                result_in_percent = result_in_price / current_division_close
                current_df_win = False
                if(goals_result == Target_or_stop_loss_return_value.TARGET_REACHED):
                    current_df_win = True
                elif(goals_result == Target_or_stop_loss_return_value.STAYED_WITHIN_LIMITS):
                    if(df.iloc[-1:]['Close'][0] > current_division_close):
                        current_df_win = True
                if(current_df_win):
                    count_success += 1
                    successes_value.append(result_in_percent)
                    percent_won += (result_in_percent)
                else:
                    count_failure += 1
                    failures_value.append(-1 * result_in_percent)
                    percent_lost -= (result_in_percent)
    if (verbose >= 1):
        total_counts = count_success + count_failure
        print("evalutated {} frames".format(total_counts))
        print("seccess-fail {}-{}. gains: {:.3f}% losses: {:.3f}%".format(count_success, count_failure,
              100 * percent_won, 100 * percent_lost))
    return count_success, count_failure, successes_value, failures_value, percent_won, percent_lost


threshold = 0.6
max_trades_per_day = 5
target_goal_percent = 0.02
stop_loss_percent = 0.01
count_success = 0
count_failure = 0
successes_value = []
failures_value = []
percent_won = 0
percent_lost = 0
verbose = 1
#print(str(datetime.datetime.now()).split('.')[0])
#for date in dates_to_check:
#    current_day_indices_with_nn_score = []
#    for i,j, score in get_indices_to_evaluate_by_date(df_rawframe_X, date, max_trades_per_day, threshold = threshold):
##        df_to_feed = get_single_df_to_feen_to_subsequent_nn(df)
##        parameter_length = len(df_to_feed.columns)
##        array_to_feed_subsequent_nn = np.array(df_to_feed, dtype=float).reshape(1, parameter_length, 1)
#        df = df_rawframe_X[i][j]
#        nn_score = score
#        total_counts = count_success + count_failure
#        if (nn_score > threshold):
#            if (verbose == 1):
#                if(total_counts % 20 == 0 and total_counts != 0):
#                    print(str(datetime.datetime.now()).split('.')[0])
#                    print("evalutated {} frames".format(total_counts))
#                    print("seccess-fail {}-{}. gains: {:.3f}% losses: {:.3f}%".format(count_success, count_failure,
#                          100 * percent_won, 100 * percent_lost))
#            
#            current_division_close = df.iloc[frame_size - 1]['Close']
#            goals_result, index_result = check_df_for_target_or_stoploss(df, target_goal_percent, stop_loss_percent, frame_size)
#            sell_price = df.iloc[index_result]['Close']
#            result_in_price = sell_price - current_division_close
#            result_in_percent = result_in_price / current_division_close
#            current_df_win = False
#            if(goals_result == Target_or_stop_loss_return_value.TARGET_REACHED):
#                current_df_win = True
#            elif(goals_result == Target_or_stop_loss_return_value.STAYED_WITHIN_LIMITS):
#                if(df.iloc[-1:]['Close'][0] > current_division_close):
#                    current_df_win = True
##                max_gain = check_df_for_maximum_gain_percent(df, frame_size)
#            if(current_df_win):
#                count_success += 1
#                successes_value.append(result_in_percent)
#                percent_won += (result_in_percent)
#            else:
#                count_failure += 1
#                failures_value.append(-1 * result_in_percent)
#                percent_lost -= (result_in_percent)

count_cached = 0
cached_evaluation_list = []
for date in dates_to_check:
    temp_lst = get_indices_to_evaluate_by_date(df_rawframe_X, date, 500, threshold = threshold)
    print(str(datetime.datetime.now()).split('.')[0])
    print("evaluated {}", len(temp_lst))
    cached_evaluation_list.append([temp_lst])
    

max_trades_per_day = 1
cached_evaluation_cropped_list = []
cached_evaluation_cropped_list_total_size = 0
for list_temp in cached_evaluation_list:
    c = 0
#    print(1)
    new_temp_list = []
    for temp_tuple in list_temp[0]:
        if (c >= max_trades_per_day):
            break
        c += 1
        new_temp_list.append(temp_tuple)
        cached_evaluation_cropped_list_total_size += 1
    cached_evaluation_cropped_list.append(new_temp_list)

#for lst in cached_evaluation_cropped_list:
#    cached_evaluation_cropped_list_total_size += len(lst)



        
threshold = 0.6
target_goal_percent = 0.026
stop_loss_percent = 0.015
verbose = 1
def print_result_for_specific_parameters(threshold, target_goal_percent, stop_loss_percent, cached_evaluation_cropped_list):
    successes,failures, s_values, v_values, p_win, p_loss = evaluate_cached_dl_list_for_reslusts(cached_evaluation_cropped_list, threshold, target_goal_percent,
                                      stop_loss_percent, verbose = 0)
    total_percent_gain = p_win - p_loss
    average_percent_gain = total_percent_gain / cached_evaluation_cropped_list_total_size
    print("----------------------------------------------------")
    print("threshold: {}, target %: {}, stop loss %: {}".format(
            threshold, 100 * target_goal_percent, 100 * stop_loss_percent))
    print("average percent gain {:.3f}% ".format(100 * average_percent_gain))


goals = np.linspace(0.02, 0.026, 13)
stops = np.linspace(0.01, 0.022, 15)
stops = np.concatenate((stops,[0.15]))
stops = np.array([0.15])

limits = []
for x in goals:
    for y in stops:
        limits.append((x,y))
for x in limits:
    print_result_for_specific_parameters(threshold, x[0], x[1], cached_evaluation_cropped_list)



def get_minute_index_of_df_when_passing_frame_size(df):
    first_minute_time = datetime.datetime.strptime('9:30:00', '%H:%M:%S')
    current_df_surpass_size_time = datetime.datetime.strptime(df.index[frame_size].split(" ")[1], '%H:%M:%S')
    return int ((current_df_surpass_size_time - first_minute_time).seconds / 60)
#    
#test_df_list = []
#for df in df_rawframe_X[2]:
#    if (len(df) > frame_size):
#        test_df_list.append(df)
#
#minutes_past_frame_size_local_list = []
#for df in test_df_list:
#    minutes_past_frame_size_local_list.append(get_minute_index_of_df_when_passing_frame_size(df))
#
#for i in range(min(minutes_past_frame_size_local_list), max(minutes_past_frame_size_local_list) + 1):
#    frames_to_check_this_minute = []
#    for j in range(len(minutes_past_frame_size_local_list)):
#        if(minutes_past_frame_size_local_list[j] == i):
#            frames_to_check_this_minute.append(j)
#    if(len(frames_to_check_this_minute) > 0):
#        print("for minute {} there are {} frames".format(i, len(frames_to_check_this_minute)))
        ############## CODE TO ACTUALLY CHECK DF ###################


#print_result_for_specific_parameters(threshold, 0.020, 0.015, cached_evaluation_cropped_list)
#print_result_for_specific_parameters(threshold, 0.021, 0.015, cached_evaluation_cropped_list)
#print_result_for_specific_parameters(threshold, 0.022, 0.015, cached_evaluation_cropped_list)
#print_result_for_specific_parameters(threshold, 0.023, 0.015, cached_evaluation_cropped_list)
#print_result_for_specific_parameters(threshold, 0.024, 0.015, cached_evaluation_cropped_list)
#print_result_for_specific_parameters(threshold, 0.025, 0.015, cached_evaluation_cropped_list)
#print_result_for_specific_parameters(threshold, 0.026, 0.015, cached_evaluation_cropped_list)

#for date in dates_to_check:
#    current_day_indices_with_nn_score = []
#    for i,j in get_indices_in_df_for_specific_date(df_rawframe_X, date):
#        
#        df = df_rawframe_X[i][j]
##        print(" i {} j {}".format(i,j))
#        if(len (df) <= frame_size):
#            continue
#        
#        df_to_feed = get_single_df_to_feen_to_subsequent_nn(df)
#        parameter_length = len(df_to_feed.columns)
#        array_to_feed_subsequent_nn = np.array(df_to_feed, dtype=float).reshape(1, parameter_length, 1)
#        nn_score = subsequent_model.predict(array_to_feed_subsequent_nn)
##        print(nn_score)
#        if (nn_score > threshold):
#            print(" i {} j {} score {}".format(i,j, nn_score[0][0]))
#            current_day_indices_with_nn_score.append((i, j, nn_score[0][0]))
#    current_day_indices_with_nn_score = pd.DataFrame(current_day_indices_with_nn_score)
#    current_day_indices_with_nn_score.columns=['i', 'j', 'score']
#    sorted_df = current_day_indices_with_nn_score.sort_values(by='score')
#    except:
#        return False


#
#goal_percentage = 0.024
#stop_loss_percentage = 0.01
#df = df_rawframe_X[7][0]
#
#goal_percentage += 1
#stop_loss_percentage = 1 - stop_loss_percentage
#
#close = df.iloc[100 - 1]['Close']
##upper_stop_price = close * goal_percentage
#df_post_division = df[100:]
#target_gain_price = close * goal_percentage
#stop_loss_price = stop_loss_percentage * close
#for i in range(len(df_post_division)):
#    line = df_post_division.iloc[i]
#    print("max ", max(line['Close'], line['Open']))
#    print("min ", min(line['Close'], line['Open']))
#    print("target" , target_gain_price)
#    print("stoploss" , stop_loss_price)
#    
#    if(max(line['Close'], line['Open']) > target_gain_price):
#        print("target reached at ", i + 100)
#        break
#    if(min(line['Close'], line['Open']) <= stop_loss_price):
#        print("stop loss at index ", i + 100)
#        break
    





#
#parameter_length = len(df_all_X.columns)
#print(5)
#    
#manual_test_train_split = True
#
#print("train test split")
#X_train = []
#Y_train = []
#X_test = []
#Y_test = []
#date_threshold_train_test = datetime.datetime.now() - timedelta(days=30)
#for i in range(len(df_all_X)):
#    if (datetime.datetime.strptime(df_all_X.index[i], '%Y-%m-%d %H:%M:%S') < date_threshold_train_test):
#        X_train.append(df_all_X.iloc[i])
#        Y_train.append(df_all_Y_binary[i])
#    else:
#        X_test.append(df_all_X.iloc[i])
#        Y_test.append(df_all_Y_binary[i])
#