# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:16:17 2020

@author: jonsnow
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import pandas_market_calendars as mcal

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

def calculate_sindlge_day_std_df(df):
    return df.drop(['Volume', 'symbol'], axis=1).std().mean()

def calculate_10day_average_volume_in_df_list(df_list, days_to_average_count = 10):
    std_list_by_dfs = []
    for df in df_list:
        std_list_by_dfs.append(calculate_sindlge_day_std_df(df))
    return float(sum(std_list_by_dfs)) / float(days_to_average_count)

def get_tickers_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry, minimum_price = 0):
    result_indices = []
    for df_sublist in df_list:
        ticker = df_sublist[0]['symbol'][0]
        try:
            close = df_sublist[0]['Close'][0]
            ticker = df_sublist[0]['symbol'][0]
            if(my_is_number(close)):
                if ((close >= minimum_boundry) and 
                    (close <= maximum_boundry)):
                    result_indices.append((ticker, close))
        except:
            pass
    return result_indices

def check_df_for_maximum_gain_percent(df, split_index):
    close_value = df['Close'][split_index-1]
    df_scan = df.iloc[split_index-1:]
    maximum_value = df_scan['High'].max()
    return maximum_value / close_value
    
def get_df_by_std_volatility_grouped_by_days(df_list, minimum_boundry, maximum_boundry):
    frame_length = 0
    X_list = []
    list_to_test = get_tickers_within_std_volatility_bondries(df_list, minimum_boundry, maximum_boundry)
    for i in range(len(list_to_test)):
        current_day_frame_length = len(df_list[list_to_test[i][0]][list_to_test[i][1]])
        if(current_day_frame_length < frame_length):
            continue
        current_df = df_list[list_to_test[i][0]][list_to_test[i][1]]
        X_list.append(current_df)
    return X_list

def my_is_number(var_to_test):
    try:
        var_to_test / 1
        if(var_to_test > 0 or var_to_test < 1):
            return True
        return False
    except:
        return False

def get_stock_data_filelist_from_folder(path_all, date_list, verbose, error_verbose):
    df_files = []
    filelist = []
    for x in date_list:
        filelist.append("{}.csv".format(x))
    for path_ticker in get_immediate_subdirectories(path_all):
        current_path = path_all+"/"+path_ticker+"/"
        tmp_list = []
        if(len(filelist) == 0):
            continue
        try:
            for filename in filelist:
                tmp_list.append((path_ticker,current_path + filename))
            df_files.append(tmp_list)
        except KeyboardInterrupt:
            break
        except:
            if(error_verbose >= 1):
                print("An exception occurred with ticker {} ".format(path_ticker))
            continue
    return df_files
    
def get_clean_data_df_from_files_grouped_by_day(files_list, verbose, error_verbose):
    DFList_all = []
    for ticker_files_list in files_list:
        DFList = []
        try:
            for ticker_file_tuple in ticker_files_list:
#                print(1)
                path_ticker, ticker_file  = ticker_file_tuple
#                print(2)
                mydata = pd.read_csv(ticker_file)
#                print(3)
                try:
                    mydata = mydata.drop(['Open Interest'], axis=1)
                except KeyboardInterrupt:
                    break
                except:
                    pass
#                print(4)
                mydata = mydata.rename(columns={'Date' : 'date'})
#                print(5)
                mydata.date = pd.to_datetime(mydata.date)
#                print(6)
                mydata['symbol'] = path_ticker
#                print(7)
                mydata = mydata.set_index('date')
#                print(8)
                DFList.append(mydata)
#                print(9)
        except KeyboardInterrupt:
            break
        except:
            if(error_verbose >= 1):
                print("An exception occurred with ticker {} function  get_clean_data_df_from_files_grouped_by_day".format(path_ticker))
            continue
        DFList_all.append(DFList)
    return DFList_all

def get_ticker_tuples_data_from_filelist(filelist, lower_limit, upper_limit, verbose, error_verbose, minimum_price = 0):
    clean_df = get_clean_data_df_from_files_grouped_by_day(filelist, verbose, error_verbose)
    ticker_tuples = get_tickers_within_std_volatility_bondries(clean_df, lower_limit, upper_limit)
    return ticker_tuples

def get_filtered_data_tickers(date_to_scan, path_stock_data, lower_limit, upper_limit, verbose, error_verbose, minimum_price = 0):
    dates_to_download = [date_to_scan]
    filelist_stock_data = get_stock_data_filelist_from_folder(path_stock_data, dates_to_download, verbose, error_verbose)
    filtered_tickers = []
    file_batch_size = 40
    
    for i in range(int(len(filelist_stock_data) / file_batch_size) + 1):
        if(verbose >= 1):
            print("loading batch {} out of {}".format(i + 1, int(len(filelist_stock_data) / file_batch_size) + 1))
        tickers = get_ticker_tuples_data_from_filelist(
                filelist_stock_data[i * file_batch_size: min((i + 1) * file_batch_size, len(filelist_stock_data))],
                lower_limit, upper_limit, verbose, error_verbose, minimum_price = minimum_price)
        for x in tickers:
            filtered_tickers.append(x)
    return filtered_tickers

def get_result_ticker_close_price_tuples(date_to_check, lower_limit, upper_limit, path_stock_data, verbose, error_verbose, minimum_price = 0):
    start_date = (datetime.strptime(date_to_check, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
    end_date = (datetime.strptime(date_to_check, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    nyse = mcal.get_calendar('NYSE')
    date_to_scan = nyse.schedule(start_date=start_date, end_date=end_date).index[-1].strftime("%Y-%m-%d")
    return  get_filtered_data_tickers(date_to_scan, path_stock_data, lower_limit, upper_limit, verbose, error_verbose, minimum_price = minimum_price)
