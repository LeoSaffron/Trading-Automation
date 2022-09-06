# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:37:06 2020

@author: jonsnow
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:50:01 2020

@author: jonsnow
"""

import fetch_single_sublist_iqfeed
import std_filter_getter
import place_simple_order_tws
import os
import sys
from datetime import datetime, timedelta
#from datetime import date
import pandas as pd
import numpy as np
import time
from _thread import start_new_thread
from time import sleep
output_tickers_folder = "stock_data/stocks_2019_2020/"
tickers_path = "tickers_finnhub_all.csv"
time_of_trade_start_str = "2020-11-25 09:30:00"
frame_size = 100
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_11_09__14_17_59"
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_11_16__02_29_17"
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_11_23__14_03_54"
path_vix_daily = "stock_data/vixcurrent_daily.csv"
path_2nd_compound_model = "2nd_model.h5"

next_check_index = 1
current_test_minute = 1
continue_threads = False
continue_threads = True


max_frames_to_buy = 5
count_stocks_buy_order_placed = 0

ideal_target_price = 500
ideal_target_price_max_lower_margin = 80

shares_to_test_with_nn = []



exchange_symbol_list_nasdaq = list(pd.read_csv("stock_data/companylist_nasdaq.csv")['Symbol'])
exchange_symbol_list_nyse = list(pd.read_csv("stock_data/companylist_nyse.csv")['Symbol'])
exchange_symbol_list_amex = list(pd.read_csv("stock_data/companylist_amex.csv")['Symbol'])

def fetch_single_timeframed_ticker_list(ticker_list, start_date, start_time, end_time):
    return fetch_single_sublist_iqfeed.main(ticker_list, start_date, start_time, end_time, 0, 0)

def test1(ticker_list, start_date, start_time, end_time):
    global running_fetching_threads, shares_to_test_with_nn
    running_fetching_threads += 1
#    print("fetching threads running: {}".format(running_fetching_threads))
    global df_list
#    print(123)
    df, faulty = fetch_single_timeframed_ticker_list(ticker_list, start_date, start_time, end_time)
    df_list.append(df)
#    print(1231)
    global end_process_time
    end_process_time = datetime.now()
#    print(len(df))
    for current_df_list in df:
        current_ticker = current_df_list['symbol'][0]
#        print(1233)
        update_df_in_tickers_dict(current_ticker, current_df_list)
        
#        print(1234)
#        print(count_ticker_df_size(tickers_dataframes_dict, current_ticker))
        if(count_ticker_df_size(tickers_dataframes_dict, current_ticker) >= frame_size):
            print("share {} reached frame size", current_ticker)
            global tickers, ticker_divided_list
            tickers.remove(current_ticker)
            for lst in ticker_divided_list:
                if current_ticker in lst:
                    lst.remove(current_ticker)
            shares_to_test_with_nn.append(current_ticker)
    for f in faulty:
        faulty_tickers.append(f)
    running_fetching_threads -= 1
#    print("fetching threads running: {}".format(running_fetching_threads))

#def minutely_process(start_time):
#    times = 0
#    global next_minute_to_check
#    while (times < 10):
#        newtime =  get_time_by_start_and_index(start_time, times)
#        
#        print("true minute index {}. past now minute {}".format(times, newtime))
##        print()
#        times += 1
#        next_minute_to_check = newtime
#        sleep(5)



def minutely_process_metronome_checker(start_time):
    times = 0
    global current_test_minute, next_minute_to_check, last_checked_minute
    global continue_threads
    while (times < 11 and continue_threads == True):
#        newtime = get_time_by_start_and_index_in_seconds(start_time, times * 18)
        newtime = datetime.now() - timedelta(hours=7)
#        print("metronome checker {}. current time {}".format(times, newtime))
        if(is_first_time_passed_certain_minute(newtime - timedelta(minutes=(0 + 0))  -  timedelta(seconds=10), current_test_minute)):
#        if(is_first_time_passed_certain_minute(newtime, current_test_minute)):
            print("just passed {} minte that has time {}".format(current_test_minute, next_minute_to_check))
#            print(1)
            
            pre_minute_str = str(last_checked_minute).split(' ')[1].replace(':', '')
#            print(2)
#            print(last_checked_minute)
            next_minute_str = str(next_minute_to_check).split(' ')[1].replace(':', '')
#            print(3)
            date_to_fetch = str(next_minute_to_check).split(' ')[0].replace('-', '')
#            print(4)
            fetch_divided_list_of_tickers(ticker_divided_list, date_to_fetch, date_to_fetch, pre_minute_str, next_minute_str, 0.085)
#            print(5)
            last_checked_minute = get_time_by_start_and_index(time_of_trade_start, current_test_minute)
#            print(6)
            current_test_minute += 10
#            print(7)
            next_minute_to_check =  get_time_by_start_and_index(time_of_trade_start, current_test_minute)
#            print(8)
        times += 1
        sleep(10)

def check_frames_for_results(ticker_list):
    for ticker in ticker_list:
        print("checked % ticker" % ticker)

#def monitor_frames_with_sufficient_length():
#    global shares_to_test_with_nn, tickers
#    while (current_test_minute < 400):
#        list_to_check = shares_to_test_with_nn.copy()
#        for item in shares_to_test_with_nn:
#            try:
#                tickers.remove(item)
#            except:
#                pass
#            print("checked {} ticker".format(item))
#    #    check_frames_for_results(list_to_check)
#        shares_to_test_with_nn = []
#        sleep(1)


def monitor_frames_with_sufficient_length():
    global shares_to_test_with_nn, tickers, potenital_gains_tickers, tickers_that_didnt_pass_filter
    global threshold
    global continue_threads
    while (current_test_minute < 400 and continue_threads == True):
        while (len(shares_to_test_with_nn) > 0):
            ticker = shares_to_test_with_nn[0]
            df_to_check = tickers_dataframes_dict[ticker]
            score_first = get_single_df_to_feen_to_first_nn(df_to_check)
            score = get_df_nn_score(df_to_check)
            newitem = {
                    'name' : ticker,
                    'nn_score' : score,
                    'trigger_time' : df_to_check.index[frame_size -1 ],
                    'discovered_time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if(score > threshold):
                potenital_gains_tickers.append(newitem)
            else:
                tickers_that_didnt_pass_filter.append(newitem)
            newitem_fisrt_nn = newitem.copy()
            newitem_fisrt_nn['nn_score'] = score_first
            if(score_first > threshold):
                potenital_gains_tickers_first_nn.append(newitem_fisrt_nn)
            else:
                tickers_that_didnt_pass_filter_first_nn.append(newitem_fisrt_nn)
            shares_to_test_with_nn.remove(ticker)
        sleep(1)


def get_minuted_time_from_full_time(full):
    pass

def is_first_time_passed_certain_minute(current_time, minute_index):
    
    return current_time >= get_time_by_start_and_index(time_of_trade_start, minute_index)

def initialize_dataframes_dict(tickers_list):
    return dict.fromkeys(tickers_list)
    
def update_df_in_tickers_dict(ticker, new_df):
    global tickers_dataframes_dict
    old_df = tickers_dataframes_dict[ticker]
    if(type(old_df) == type(None)):
        tickers_dataframes_dict[ticker] = new_df.copy()
    else:
        tickers_dataframes_dict[ticker] = pd.concat([old_df, new_df])




def select_tickers_from_list(ticker_list, limit):
    return ticker_list[:min(limit, len(ticker_list))].copy()


def fetch_divided_list_of_tickers(divided_list, start_date, end_date, start_time, end_time, delay):
    for lst in ticker_divided_list:
        start_new_thread(test1, (lst, start_date, start_time, end_time))
        time.sleep(delay)

def transform_time_into_destination_time():
    pass

def get_time_by_start_and_index(start_time, minute_index):
    return start_time + timedelta(minutes=minute_index)

def get_time_by_start_and_index_in_seconds(start_time, second_index):
    return start_time + timedelta(seconds=second_index)

def count_ticker_df_size(df_dict, ticker):
    df = df_dict[ticker]
    if(type(df) == type(None)):
        return 0
    return len(df)



import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import tensorflow.keras.backend
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import model_from_json, load_model
from keras.layers import Dropout
from keras.utils import to_categorical
vix_daily_min = 8.56
vix_daily_mean = 10.323445
colse_divided_by_open_min = 0.5
close_divided_by_subframe_min = 0.8
close_divided_by_subframe_factor = 2.5
threshold = 0.5
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
def get_vix_fdaily_from_file_and_scale(path, min_value, mean):
    vix_daily = pd.read_csv(path, index_col='Date')
    vix_daily = vix_daily - min_value
    vix_daily = vix_daily / (2 * mean)
    return vix_daily

def df_cast_type(df):
    df['Open'] = df['Open'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)

def get_close_divided_by_max_in_single_df(df):
    min_value = df['Low'].min()
    max_value = df['High'].max() - min_value
    close_value = df['Close'][-1:][0] - min_value
    return close_value / max_value

def get_close_divided_by_open_in_single_df(df):
    open_value = df['Open'].max()
    close_value = df['Close'][-1:][0]
    return close_value / open_value

def get_close_divided_by_open_subframe_in_single_df(df, subframe_length = 30):
    open_subframe_value = df['Close'][-31:-subframe_length][0]
    close_value = df['Close'][-1:][0]
    return close_value / open_subframe_value


def get_primary_exchange_by_symbol(symbol):
    if(symbol in exchange_symbol_list_nasdaq):
        return "NASDAQ"
    if(symbol in exchange_symbol_list_nyse):
        return "NYSE"
    if(symbol in exchange_symbol_list_nasdaq):
        return "AMEX"
    return None





###########################
##### SCALE AND RERRANGE
###########################

def add_bollinger_bands_td_df_list(df_to_scale):
    period = 20
#    multiplier = 2
    df = df_to_scale
    df['UpperBand_Width2'] = df['Close'].rolling(period).mean() + df['Close'].rolling(period).std() * 2
    df['LowerBand_Width2'] = df['Close'].rolling(period).mean() - df['Close'].rolling(period).std() * 2
    df['UpperBand_Width3'] = df['Close'].rolling(period).mean() + df['Close'].rolling(period).std() * 3
    df['LowerBand_Width3'] = df['Close'].rolling(period).mean() - df['Close'].rolling(period).std() * 3
    df = df.fillna(0)
    return df

def scale_and_rearrange_df_list(df_to_scale):
    df = np.array(df_to_scale)
    high = df.T[0].max()
    low = df.T[1].min()
    delta = high - low
    high_col = (df.T[0] - low) / delta
    low_col = (df.T[1] - low) / delta
    open_col = (df.T[2] - low) / delta
    close_col = (df.T[3] - low) / delta
    
    bollinger_band_width_2_upper = (df.T[5] - low) / delta
    bollinger_band_width_2_lower = (df.T[6] - low) / delta
    bollinger_band_width_3_upper = (df.T[7] - low) / delta
    bollinger_band_width_3_lower = (df.T[8] - low) / delta
    
    high_volume = df.T[4].max()
    low_volume = df.T[4].min()
    delta_volume = high_volume - low_volume
    volume_col = (df.T[4] - low_volume) / delta_volume
    
#    df_all_X[i] = np.array([high_col, low_col, open_col, close_col, volume_col]).T
    
    padding_col = np.zeros(df.shape[0])
    df = np.array([padding_col, bollinger_band_width_3_upper, bollinger_band_width_2_upper, high_col, high_col, close_col, close_col, open_col, open_col,
            low_col, low_col, bollinger_band_width_2_lower, bollinger_band_width_3_lower, padding_col, volume_col, volume_col, padding_col]).T
    return df

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
    high = float(df.drop(['Volume'], axis=1)['High'].max())
    low = float(df.drop(['Volume'], axis=1)['Low'].min())
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

def trasfroem_df_to_feed_to_trivial_cnn_old(df_to_transform, frame_size):
    transformed_df = scale_df_for_trivial_cnn_dataframe(df_to_transform[:frame_size])
    transformed_df = duplicate_columns_in_df_for_trivial_cnn_dataframe(transformed_df)
    transformed_df = switch_column_positions_in_df_for_trivial_cnn_dataframe(transformed_df)
    transformed_df = add_padding_in_df_for_trivial_cnn_dataframe(transformed_df) 
    transformed_df = np.array(transformed_df)
    transformed_df = transformed_df.reshape((1, frame_size, 10, 1))
    return transformed_df

def trasfroem_df_to_feed_to_trivial_cnn(df_to_transform, frame_size):
    transformed_df = (add_bollinger_bands_td_df_list(df_to_transform[:frame_size])).drop('symbol', axis=1)
    transformed_df = scale_and_rearrange_df_list(transformed_df)
    frame_width = transformed_df.shape[1]
    transformed_df = transformed_df.reshape((1, frame_size, frame_width, 1))
    return transformed_df


def get_single_df_to_feen_to_first_nn(df):
    global vix_today
    df_cast_type(df)
#    df_cropped = df[:frame_size].copy()
    transformed_df = trasfroem_df_to_feed_to_trivial_cnn(df, frame_size)
#    X_train = X_train.reshape(X_train.shape[0], parameter_length, 1)
    prediction_trivial_cnn = model_trivial_dnn.predict(transformed_df)[0][0]
    return prediction_trivial_cnn


def get_single_df_to_feen_to_subsequent_nn(df):
    global vix_today
    df_cast_type(df)
    df_cropped = df[:frame_size].copy()
    transformed_df = trasfroem_df_to_feed_to_trivial_cnn(df, frame_size)
#    X_train = X_train.reshape(X_train.shape[0], parameter_length, 1)
    prediction_trivial_cnn = model_trivial_dnn.predict(transformed_df)[0][0]
    current_df_date = df_cropped.index[0]
    current_df_volatility = df_cropped.drop('Volume', axis =1).std().mean() * 3
    current_df_vix = vix_today
    current_df_close_percent_of_max = get_close_divided_by_max_in_single_df(df_cropped)
    current_df_close_percent_of_open = get_close_divided_by_open_in_single_df(df_cropped)
    current_df_close_percent_of_open_subframe = get_close_divided_by_open_subframe_in_single_df(df_cropped, subframe_length = 30)
    df_to_feed = pd.DataFrame([current_df_date, prediction_trivial_cnn, current_df_volatility, current_df_vix,
                         current_df_close_percent_of_max, current_df_close_percent_of_open, current_df_close_percent_of_open_subframe]).T
    df_to_feed.columns = ['date', 'first_nn_score', 'volatility', 'vix', 'close_percent_of_max',
                    'close_percent_of_open', 'close_percent_of_open_subframe']
    df_to_feed = df_to_feed.set_index('date')
    return df_to_feed

def get_df_nn_score(df):
    df_to_feed = get_single_df_to_feen_to_subsequent_nn(df)
    parameter_length = len(df_to_feed.columns)
    array_to_feed_subsequent_nn = np.array(df_to_feed, dtype=float).reshape(1, parameter_length, 1)
    nn_score = subsequent_model.predict(array_to_feed_subsequent_nn)
    return nn_score[0][0]



def get_percentage_goal_by_nn_score(score):
    if (score > 0.80):
        return 1.03
    if (score > 0.70):
        return 1.024
    if (score > 0.60):
        return 1.02
    if (score > 0.50):
        return 1.015
    return 1.01

def get_percentage_list_by_score_for_df(df):
    column = df['nn_score']
    result_list = []
    for item in column:
        result_list.append(get_percentage_goal_by_nn_score(item))
    return result_list

def get_goal_single_share_price_list_by_score_for_df(df):
    result_list = []
    for row_index in range(len(df)):
        result_row = round(float( tickers_dataframes_dict[df.iloc[row_index]['name']][-1:]['Close']) * df.iloc[row_index]['goal_percentage'] - 0.01, 2)
        result_list.append(result_row)
    return result_list

order_tickers_buy_order_sent = []
order_tickers_buy_order_failed = []
order_tickers_below_threshold = []
order_tickers_not_sent_due_higher_score = []

def action_on_symbol(df, index):
    row = df.iloc[index]
    global order_tickers_buy_order_failed
    try:
        ticker = row['name']
        primary_exchange = get_primary_exchange_by_symbol(ticker)
        if (not type(primary_exchange) == type(None)):
            print("buying stock {}".format(row['name']))
            price, quantity = get_price_and_quantity_for_single_row(df, index)
            print("buying stock {}. price {}. quantity {}".format(row['name'], price, quantity))
            place_simple_order_tws.place_order(ticker, primary_exchange, quantity, price, "BUY")
            
            global count_stocks_buy_order_placed
            count_stocks_buy_order_placed += 1
            global order_tickers_buy_order_sent
            order_tickers_buy_order_sent.append(row)
        else:
#            global order_tickers_buy_order_failed
            order_tickers_buy_order_failed.append(row)
    except:
#        global order_tickers_buy_order_failed
        order_tickers_buy_order_failed.append(row)

    

def get_latest_price_by_name(ticker):
    max_fetching_trials = 4
    result = -1
    for x in range((max_fetching_trials)):
        try:
            fetching_result = fetch_single_sublist_iqfeed.main([ticker], time_of_trade_start_str.split(' ')[0].replace('-',''), "103000", "160000", 0, 0)
            result = fetching_result[0][0][-1:]['Close']
            break
        except:
            sleep(1)
    return float(result)

def get_price_and_quantity_for_single_row(df, index):
    fetched_price = get_latest_price_by_name(df.iloc[index]['name'])
    if(fetched_price <= 0):
        return -1
    price = round(float(fetched_price) +0.01, 2)
    quantity = int(ideal_target_price / price)
    if(ideal_target_price - quantity * price <= ideal_target_price_max_lower_margin):
        quantity += 1
    return price, quantity

#def get_amount_by_price_list_by_score_for_df(df):
#    fetched_price = get_latest_price_by_name(df.iloc[index]['name'])
#    if(fetched_price <= 0):
#        return -1
#    price = round(float(fetched_price) +0.01, 2)
#    quantity = int(ideal_target_price / price)
#    if(ideal_target_price - quantity * price <= ideal_target_price_max_lower_margin):
#        quantity += 1
#    return quantity

def place_buy_orders_for_dict():
    global max_frames_to_buy, count_stocks_buy_order_placed
    global potenital_gains_tickers_first_nn, potenital_gains_tickers_first_nn_processed
    potenital_gains_tickers_first_nn_copy = potenital_gains_tickers_first_nn.copy()
    potenital_gains_tickers_first_nn = []
    max_frames_to_buy_current  = max_frames_to_buy - count_stocks_buy_order_placed
    max_frames_to_buy_current  = min(max_frames_to_buy_current, len(potenital_gains_tickers_first_nn_copy))
    scored_df = pd.DataFrame(potenital_gains_tickers_first_nn_copy).sort_values(by=['nn_score'], ascending=False)
    scored_df = scored_df[:min(max_frames_to_buy_current, len(scored_df))]
    scored_df['goal_percentage'] = get_percentage_list_by_score_for_df(scored_df)
    scored_df['goal_price'] = get_goal_single_share_price_list_by_score_for_df(scored_df)
    for i in range((min(3, max_frames_to_buy_current))):
        action_on_symbol(scored_df, i)
    if((max_frames_to_buy_current > 3)):
        for i in range(3, (min(4, max_frames_to_buy_current))):
            if(scored_df.iloc[i]['nn_score'] >= 0.6):
                action_on_symbol(scored_df, i)
            else:
                order_tickers_below_threshold.append(scored_df.iloc[i])
    if((max_frames_to_buy_current > 4)):
        for i in range(4, (min(5, max_frames_to_buy_current))):
            if(scored_df.iloc[i]['nn_score'] >= 0.7):
                action_on_symbol(scored_df, i)
            else:
                order_tickers_below_threshold.append(scored_df.iloc[i])
    for i in range(max_frames_to_buy_current, len(scored_df)):
        order_tickers_not_sent_due_higher_score.append(scored_df.iloc[i])
    potenital_gains_tickers_first_nn_processed.append(potenital_gains_tickers_first_nn_copy)
    
def monitor_stocks_ready_to_buy():
    global continue_threads, current_test_minute
    global potenital_gains_tickers_first_nn
    while (current_test_minute < 400 and continue_threads == True):
        global potenital_gains_tickers_first_nn
        if(len(potenital_gains_tickers_first_nn) > 0):
            sleep(10)
            place_buy_orders_for_dict()
#            sleep(20)
        sleep(1)

#tickers_path = "tickers_finnhub_all.csv"
#tickers_prefiltered = pd.read_csv(tickers_path).drop(['0'],axis=1).T.iloc[0]

#pd.DataFrame(results).to_csv("daily_results.csv", index=False)
    

#results_temp = pd.read_csv("daily_results.csv")
#results = []
#for i in range(len(results_temp)):
#    result = results_temp.iloc[i]
#    results.append((result[0], result[1]))


time_of_trade_start = datetime.strptime(time_of_trade_start_str, "%Y-%m-%d %H:%M:%S")
next_minute_to_check =  get_time_by_start_and_index(time_of_trade_start, current_test_minute)
last_checked_minute = time_of_trade_start
s = datetime.now()
results = std_filter_getter.get_result_ticker_std_tuples(time_of_trade_start_str.split(' ')[0], 0.3, 3.0, "stock_data/stocks_2019_2020", 0, 0, minimum_price = 7)
e = datetime.now()
print ( e - s )
max_amount_to_monitor = 11700
running_fetching_threads = 0
tickers_all  = list(np.array(results).T[0])
tickers = select_tickers_from_list(tickers_all, max_amount_to_monitor)
tickers_dataframes_dict = initialize_dataframes_dict(tickers)
amount_to_divide = 150
amount_to_divide = min(amount_to_divide, len(tickers))
divion_indices = np.linspace(0,len(tickers), amount_to_divide+1, dtype=int)
faulty_tickers = []
ticker_divided_list = []
df_list = []
potenital_gains_tickers = []
potenital_gains_tickers_first_nn = []
potenital_gains_tickers_first_nn_processed = []
tickers_that_didnt_pass_filter_first_nn = []
tickers_that_didnt_pass_filter = []
vix_daily = get_vix_fdaily_from_file_and_scale(path_vix_daily, vix_daily_min, vix_daily_mean)
#vix_today = vix_daily.loc[time_of_trade_start.strftime("%Y-%m-%d")][0]
vix_today = vix_daily.loc[time_of_trade_start.strftime("%m/%d/%Y")][0]
for i in range(amount_to_divide):
#    print(divion_indices[i], divion_indices[i+1])
    ticker_divided_list.append(tickers[divion_indices[i] : divion_indices[i + 1]])
#tickers = tickers[tickers[tickers == 'WHD'].index[0]+1:]
#start_process_time = datetime.now()
#end_process_time = datetime.now()
#fetch_divided_list_of_tickers(ticker_divided_list, '20201020', '20201020', '093000', '100000', 0.035)
#fetch_divided_list_of_tickers(ticker_divided_list, '20201020', '20201020', '100000', '103000', 0.035)

#
#for i in range(0):
#    tickers = faulty_tickers.copy()
##    global faulty_tickers
#    faulty_tickers = []
#    ticker_divided_list = []
#    amount_to_divide = 10
#    amount_to_divide = min(amount_to_divide, len(tickers))
#    divion_indices = np.linspace(0,len(tickers), amount_to_divide+1, dtype=int)
#    for i in range(amount_to_divide):
#        ticker_divided_list.append(tickers[divion_indices[i] : divion_indices[i + 1]])
#        fetch_divided_list_of_tickers(ticker_divided_list, '20201020', '20201020', '093000', '100000', 0.015)
##
#
l = 0
for current_df_list in df_list:
    l += len(current_df_list)

#
#end_process_time - start_process_time


#start_new_thread(minutely_process, (time_of_trade_start,))
start_process_time = datetime.now()
end_process_time = datetime.now()
start_new_thread(minutely_process_metronome_checker, (time_of_trade_start,))

start_new_thread(monitor_frames_with_sufficient_length, ())
start_new_thread(monitor_stocks_ready_to_buy, ())






#df_to_check = tickers_dataframes_dict['IBIO'][:frame_size]
#vix_df = vix_today
#nn_score = score



    
    

#potenital_gains_tickers = []
#threshold = 0.7
#testlist = shares_to_test_with_nn[:30]
#s = datetime.now()
#for ticker in testlist:
#    df_to_check = tickers_dataframes_dict[ticker]
#    score = get_df_nn_score(df_to_check)
#    if(score > threshold):
#        newitem = {
#                'name' : ticker,
#                'nn_score' : score,
#                'trigger_time' : df_to_check.index[frame_size -1 ],
#                'discovered_time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#        }
#        potenital_gains_tickers.append(newitem)
#e = datetime.now()


#testlist = shares_to_test_with_nn[:30]
#s = datetime.now()
#while (len(testlist) > 0):
#    ticker = testlist[0]
#    df_to_check = tickers_dataframes_dict[ticker]
#    score = get_df_nn_score(df_to_check)
#    if(score > threshold):
#        newitem = {
#                'name' : ticker,
#                'nn_score' : score,
#                'trigger_time' : df_to_check.index[frame_size -1 ],
#                'discovered_time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#        }
#        potenital_gains_tickers.append(newitem)
#    testlist.remove(ticker)
#e = datetime.now()

#
#for current_df_list in df_list:
#    for df in current_df_list:
#        sym = df['symbol'][0]
#        df_new = df.drop('symbol', axis=1)
#        outdir = output_tickers_folder + sym
#    #        print(outdir)
#        if not os.path.exists(outdir):
#            os.mkdir(outdir)
#        output_tickers_folder + "{}/{}.csv".format(sym,date.today().strftime("%Y_%m_%d"))
#    #            filename = "%s.csv" % sym
#        filename = output_tickers_folder + "{}/{}.csv".format(sym,date.today().strftime("%Y_%m_%d"))
#        df_new.to_csv(filename)
#start_new_thread(factorial, (4, ))
#
#output_dataframes_folder = "stock_data/test_data6/"
#all_df = []
#for x in df_list:
#    for item in x:
#        all_df.append(item)
#
##i = 0
##for df in all_df:
##    newdf = []
##    print(i)
##    i +=1
##    for groupyear in df.groupby(pd.to_datetime(df.index).year):
##        for groupmonth in groupyear[1].groupby(pd.to_datetime(groupyear[1].index).month):
##            for group in groupmonth[1].groupby(pd.to_datetime(groupmonth[1].index).day):
##                newdf.append(group[1])
##    for df_sub in newdf:
##        ticker = df_sub['symbol'][0]
###        frame_date = str(df_sub.index[0].date())
##        frame_date = str(df_sub.index[0]).split(' ')[0]
##        try:
##            outdir_temp = output_dataframes_folder  + str(ticker) + '/'
##            if not os.path.exists(outdir_temp):
##                os.mkdir(outdir_temp)
##            output_file_path = outdir_temp + frame_date + ".csv"
##            df_sub.to_csv(output_file_path)
##        except:
##            print( "Unexpected error:", sys.exc_info()[0])
##            print(outdir_temp)
##            print( "Error in ticker {}",ticker)
#    
#print("Waiting for threads to return...")
#sleep(waiting)