# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:50:01 2020

@author: jonsnow
"""

import fetch_single_sublist_iqfeed
import os
import sys
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import time
from _thread import start_new_thread
from time import sleep

output_tickers_folder = "stock_data/stocks_2019_2020/"
tickers_path = "tickers_finnhub_all.csv"
date_to_fetch_from = '20210201'


#df_list = fetch_single_sublist_iqfeed.main(['TSLA', 'JNUG', 'NKLA'], '20201016', '100000', '101500')


def fetch_single_timeframed_ticker_list(ticker_list, start_date, start_time, end_time):
    return fetch_single_sublist_iqfeed.main(ticker_list, start_date, start_time, end_time, 0, 0)

df_list = []
def test1(ticker_list, start_date, start_time, end_time):
    global running_fetching_threads
    running_fetching_threads += 1
    print("fetching threads running: {}".format(running_fetching_threads))
    global df_list
    df, faulty = fetch_single_timeframed_ticker_list(ticker_list, start_date, start_time, end_time)
    df_list.append(df)
    global end_process_time
    end_process_time = datetime.now()
    
    for f in faulty:
        faulty_tickers.append(f)
    running_fetching_threads -= 1
    print("fetching threads running: {}".format(running_fetching_threads))
    
faulty_tickers = []
#list1 = ['TSLA', 'JNUG', 'NKLA']
#list2 = ['AA', 'CITI']
running_fetching_threads = 0




tickers_path = "tickers_finnhub_all.csv"
#    output_tickers_folder = "stock_data/fetched_by_iqfeed/"
faulty_tickers = []
ticker_divided_list = []
tickers = pd.read_csv(tickers_path).drop(['0'],axis=1).T.iloc[0]
amount_to_divide = 150
divion_indices = np.linspace(0,len(tickers), amount_to_divide+1, dtype=int)
for i in range(amount_to_divide):
#    print(divion_indices[i], divion_indices[i+1])
    ticker_divided_list.append(tickers[divion_indices[i] : divion_indices[i + 1]])
#tickers = tickers[tickers[tickers == 'WHD'].index[0]+1:]
start_process_time = datetime.now()
end_process_time = datetime.now()
for lst in ticker_divided_list:
    start_new_thread(test1, (lst, date_to_fetch_from, '093000', '160000'))
    time.sleep(0.335)

time.sleep(3 * 60)

for i in range(10):
    tickers = faulty_tickers.copy()
    faulty_tickers = []
    ticker_divided_list = []
    amount_to_divide = 50
    divion_indices = np.linspace(0,len(tickers), amount_to_divide+1, dtype=int)
    for i in range(amount_to_divide):
        ticker_divided_list.append(tickers[divion_indices[i] : divion_indices[i + 1]])
    #tickers = tickers[tickers[tickers == 'WHD'].index[0]+1:]
    #start_process_time = datetime.now()
    #end_process_time = datetime.now()
    for lst in ticker_divided_list:
        start_new_thread(test1, (lst, date_to_fetch_from, '093000', '160000'))
        time.sleep(0.335)
    time.sleep(2 * 60)


l = 0
for current_df_list in df_list:
    l += len(current_df_list)


end_process_time - start_process_time
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

#output_dataframes_folder = "stock_data/output_tickers_folder/"
output_dataframes_folder = output_tickers_folder
all_df = []
for x in df_list:
    for item in x:
        all_df.append(item)

i = 0
for df in all_df:
    newdf = []
    print(i)
    i +=1
    for groupyear in df.groupby(pd.to_datetime(df.index).year):
        for groupmonth in groupyear[1].groupby(pd.to_datetime(groupyear[1].index).month):
            for group in groupmonth[1].groupby(pd.to_datetime(groupmonth[1].index).day):
                newdf.append(group[1])
    for df_sub in newdf:
        ticker = df_sub['symbol'][0]
#        frame_date = str(df_sub.index[0].date())
        frame_date = str(df_sub.index[0]).split(' ')[0]
        try:
            outdir_temp = output_dataframes_folder  + str(ticker) + '/'
            if not os.path.exists(outdir_temp):
                os.mkdir(outdir_temp)
            output_file_path = outdir_temp + frame_date + ".csv"
            df_sub.to_csv(output_file_path)
        except:
            print( "Unexpected error:", sys.exc_info()[0])
            print(outdir_temp)
            print( "Error in ticker {}",ticker)
    
print("Waiting for threads to return...")
sleep(waiting)