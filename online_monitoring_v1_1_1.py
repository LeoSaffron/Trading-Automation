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
time_of_trade_start_str = "2021-01-22 09:30:00"
frame_size = 100
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_11_09__14_17_59"
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_11_16__02_29_17"
path_trivial_cnn = "measured_stats/models_results/trivial_cnn/2020_11_23__14_03_54"
path_vix_daily = "stock_data/vixcurrent_daily.csv"
#path_2nd_compound_model = "2nd_model.h5"

next_check_index = 1
current_test_minute = 1
continue_threads = False
continue_threads = True


max_frames_to_buy = 5
count_stocks_buy_order_placed = 0

ideal_target_price = 500
ideal_target_price_max_lower_margin = 80

shares_to_test_with_nn = []

arr_monitor_mins_to_check = [50,30,10,5]
arr_monitor_mins_to_wait_after_check = [20,20,20,10]
arr_monitor_checked_index = 0
mins_to_check_standard = 1
mins_to_check_delay = 1


exchange_symbol_list_nasdaq = list(pd.read_csv("stock_data/companylist_nasdaq.csv")['Symbol'])
exchange_symbol_list_nyse = list(pd.read_csv("stock_data/companylist_nyse.csv")['Symbol'])
exchange_symbol_list_amex = list(pd.read_csv("stock_data/companylist_amex.csv")['Symbol'])




def get_delta_minute_to_check():
    if (arr_monitor_checked_index < len(arr_monitor_mins_to_check)):
        return arr_monitor_mins_to_check[arr_monitor_checked_index]
    return mins_to_check_standard

def get_delta_minute_delay():
    global arr_monitor_checked_index
    arr_monitor_checked_index += 1
    if (arr_monitor_checked_index <= len(arr_monitor_mins_to_check)):
        return arr_monitor_mins_to_wait_after_check[arr_monitor_checked_index - 1]
    return mins_to_check_delay

def fetch_single_timeframed_ticker_list(ticker_list, start_date, start_time, end_time):
    return fetch_single_sublist_iqfeed.main(ticker_list, start_date, start_time, end_time, 0, 0)

def test1(ticker_list, start_date, start_time, end_time, verbose = 0):
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
            if ( verbose >= 1):
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



def minutely_process_metronome_checker(start_time, verbose = 0):
    times = 0
    global current_test_minute, next_minute_to_check, last_checked_minute
    global continue_threads
    while (times < 91111 and continue_threads == True):
#        newtime = get_time_by_start_and_index_in_seconds(start_time, times * 18)
        newtime = datetime.now() - timedelta(hours=7)
#        print("metronome checker {}. current time {}".format(times, newtime))
        if(is_first_time_passed_certain_minute(newtime - timedelta(minutes=(0 + 0))  -  timedelta(seconds=2), current_test_minute)):
#        if(is_first_time_passed_certain_minute(newtime, current_test_minute)):
            if ( verbose >= 1):
                print("just passed {} minte that has time {}".format(current_test_minute, next_minute_to_check))
            pre_minute_str = str(last_checked_minute).split(' ')[1].replace(':', '')
            next_minute_str = str(next_minute_to_check).split(' ')[1].replace(':', '')
            date_to_fetch = str(next_minute_to_check).split(' ')[0].replace('-', '')
            fetch_divided_list_of_tickers(ticker_divided_list, date_to_fetch, date_to_fetch, pre_minute_str, next_minute_str, 0.185)
            last_checked_minute = get_time_by_start_and_index(time_of_trade_start, current_test_minute)
#            current_test_minute += 1
            current_test_minute += get_delta_minute_to_check()
            next_minute_to_check =  get_time_by_start_and_index(time_of_trade_start, current_test_minute)
            sleep(get_delta_minute_delay())
        times += 1
        sleep(10)

#def check_frames_for_results(ticker_list):
#    for ticker in ticker_list:
#        print("checked % ticker" % ticker)


def monitor_frames_with_sufficient_length():
    global shares_to_test_with_nn, tickers, potenital_gains_tickers, tickers_that_didnt_pass_filter
    global threshold, max_score
    global continue_threads
    while (current_test_minute < 400 and continue_threads == True):
        while (len(shares_to_test_with_nn) > 0):
            ticker = shares_to_test_with_nn[0]
            df_to_check = tickers_dataframes_dict[ticker]
            score = get_single_df_to_feen_to_first_nn(df_to_check)
            newitem = {
                    'name' : ticker,
                    'nn_score' : score,
                    'trigger_time' : df_to_check.index[frame_size -1 ],
                    'discovered_time' : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if((score > threshold) and (score <= max_score)):
                potenital_gains_tickers_first_nn.append(newitem)
            else:
                tickers_that_didnt_pass_filter_first_nn.append(newitem)
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
threshold = 0.6
max_score = 0.7
model_trivial_dnn = keras.models.load_model(
    path_trivial_cnn +"/model.h5",
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
    df_cast_type(df)
    transformed_df = trasfroem_df_to_feed_to_trivial_cnn(df, frame_size)
    prediction_trivial_cnn = model_trivial_dnn.predict(transformed_df)[0][0]
    return prediction_trivial_cnn

def get_percentage_goal_by_nn_score(score):
    return 1.028
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
order_tickers_sell_order_sent = []

orders_sent_tws = []
orders_sent_sell_tws = []

def action_on_symbol(df, index):
    row = df.iloc[index].copy()
    global order_tickers_buy_order_failed
    global orders_sent_tws
    try:
        ticker = row['name']
        primary_exchange = get_primary_exchange_by_symbol(ticker)
        if (not type(primary_exchange) == type(None)):
            print("buying stock {}".format(row['name']))
            price, quantity = get_price_and_quantity_for_single_row(df, index)
            print("buying stock {}. price {}. quantity {}".format(row['name'], price, quantity))
#            orders_sent_tws.append(place_simple_order_tws.place_order(ticker, primary_exchange, quantity, "BUY", price=price))
            orders_sent_tws.append(place_simple_order_tws.place_order(ticker, primary_exchange, quantity, "BUY", order_type="MKT"))
            
            row['quantity'] = quantity
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



def place_sell_order_on_symbol(row):
#    row = df.iloc[index]
    global order_tickers_buy_order_failed
    global orders_sent_tws
    try:
        ticker = row['name']
        price = row['goal_price']
        quantity = row['quantity']
        primary_exchange = get_primary_exchange_by_symbol(ticker)
        if (not type(primary_exchange) == type(None)):
            print("selling stock {}".format(row['name']))
#            price, quantity = get_price_and_quantity_for_single_row(df, index)
            print("selling stock {}. price {}. quantity {}".format(row['name'], price, quantity))
            orders_sent_tws.append(place_simple_order_tws.place_order(ticker, primary_exchange, quantity, "SELL", price=price))
#            global count_stocks_buy_order_placed
#            count_stocks_buy_order_placed += 1
#            global order_tickers_buy_order_sent
#            order_tickers_buy_order_sent.append(row)
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


def monitor_stocks_to_place_sell_order():
    global order_tickers_buy_order_sent, order_tickers_sell_order_sent, orders_sent_tws
    while (current_test_minute < 400 and continue_threads == True):
        if(len(order_tickers_buy_order_sent) > 0):
            sleep(25)
            orders_to_send_temp = []
            for x in order_tickers_buy_order_sent : 
                order_tickers_sell_order_sent.append(x) 
                orders_to_send_temp.append(x)
            order_tickers_buy_order_sent = []
            for stock_to_sell in orders_to_send_temp:
                orders_sent_sell_tws.append(place_sell_order_on_symbol(stock_to_sell))
        sleep(1)
    
def monitor_stocks_ready_to_buy():
    global continue_threads, current_test_minute
    global potenital_gains_tickers_first_nn
    while (current_test_minute < 180 and continue_threads == True):
        global potenital_gains_tickers_first_nn
        if(len(potenital_gains_tickers_first_nn) > 0):
            sleep(10)
            place_buy_orders_for_dict()
#            sleep(20)
        sleep(1)

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
for i in range(amount_to_divide):
    ticker_divided_list.append(tickers[divion_indices[i] : divion_indices[i + 1]])

length_ticker_list = 0
for current_df_list in df_list:
    length_ticker_list += len(current_df_list)
print("scanning {} stocks".format(length_ticker_list))



#start_new_thread(minutely_process, (time_of_trade_start,))
start_process_time = datetime.now()
end_process_time = datetime.now()
start_new_thread(minutely_process_metronome_checker, (time_of_trade_start,))

start_new_thread(monitor_frames_with_sufficient_length, ())
start_new_thread(monitor_stocks_ready_to_buy, ())
start_new_thread(monitor_stocks_to_place_sell_order, ())


