# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:03:43 2020

@author: jonsnow
"""


import fetch_single_sublist_iqfeed
import std_filter_getter


def get_daily_tickers_after_filtered_by_std():
    results = std_filter_getter.get_result_ticker_std_tuples("2020-10-16", 0.01, 0.05, "stock_data/iqfeed_last_months2", 0, 0)

def fetch_timeframed_tickers():
    pass

def fetch_single_timeframed_ticker_list(ticker_list, start_date, start_time, end_time):
    return fetch_single_sublist_iqfeed.main(ticker_list, start_date, start_time, end_time)

def fetch_single_timeframed_ticker_list_in_a_thread(ticker_list, start_date, start_time, end_time):
    ##add thread
    pass
    return fetch_single_sublist_iqfeed.main(ticker_list, start_date, start_time, end_time)

def combine_df_with_new_data():
    pass

def get_top_tickers():
    pass

def check_df_for_requierements():
    pass

def predic_df_for_gain():
    pass