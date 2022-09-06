# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 23:00:06 2020

@author: jonsnow
"""

import requests
import numpy as np
import pandas as pd
import requests
import time

api_key = "btno8p748v6p0j27h5l0"
sleep_bertween_requests = 1

r = requests.get('https://finnhub.io/api/v1/stock/symbol?exchange=US&token=' + api_key)
#print(r.json())

tickers_stocks = [item for item in r.json() if item['type'] == 'EQS']
tickers_stocks = [item for item in tickers_stocks if item['currency'] == 'USD']
#tickers_stocks_clean = []
#for item in tickers_stocks:
#    tickers_stocks_clean.append(item['symbol'])
#################### WRITE ALL TICKERS TO FILE
#pd.DataFrame(tickers_stocks)['symbol'].to_csv('tickers_finnhub_all')
tickers_stocks_clean = pd.DataFrame(pd.DataFrame(tickers_stocks)['symbol'])

stock_data = tickers_stocks_clean.set_index('symbol')
stock_data['market_cap'] = ''
stock_data['screened_for_market_cap'] = False
#for current_ticker in tickers_stocks_clean:
#    r = requests.get('https://finnhub.io/api/v1/stock/profile2?symbol=' + current_ticker + '&token=' + api_key)
#    r.json()['marketCapitalization']

count = 0
max_count = len(stock_data)
for current_ticker in stock_data.T:
    try:
#        print(x)

        r = requests.get('https://finnhub.io/api/v1/stock/profile2?symbol=' + current_ticker + '&token=' + api_key)
        stock_data.ix[current_ticker, 'market_cap'] = r.json()['marketCapitalization']
        count += 1
        if (count % 10 == 0):
            print("scraped {} out of {}".format(count, max_count))
        stock_data.ix[current_ticker, 'screened_for_market_cap'] = True
        time.sleep(sleep_bertween_requests)
    except KeyboardInterrupt:
        break
    except:
        print("An exception occurred") 
        count +=1
#        time.sleep(10)    
stock_data.to_csv("tickers_with_market_cap.csv")
print(r.json())

for current_ticker in stock_data.T:
    try:
        if(stock_data.loc[current_ticker]['screened_for_market_cap'] == False):
            print(current_ticker)
            r = requests.get('https://finnhub.io/api/v1/stock/profile2?symbol=' + current_ticker + '&token=' + api_key)
            stock_data.ix[current_ticker, 'market_cap'] = r.json()['marketCapitalization']
            count += 1
            if (count % 10 == 0):
                print("scraped {} out of {}".format(count, max_count))
            stock_data.ix[current_ticker, 'screened_for_market_cap'] = True
            time.sleep(sleep_bertween_requests)
            
    except KeyboardInterrupt:
        break
    except:
        print("An exception occurred") 
        count +=1
        time.sleep(10)    
            