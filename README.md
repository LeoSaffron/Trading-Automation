# Trading-automation
This is my attempt to create a trading bot for the stock market

## Abstract
As most of the algorythmic trading done today is HFT my goal is to develop a bot that can utilize ML in order to trade stocks like a regular retail investor.  
The trading is mostly done on midcap stocks, as the criteria for which stock to filter was chosen thanks to the analysis performed in this project

## Approach for building the model
Algotrading algos usually look at the stock market data as time series and then treat it appropriately.  
I chose a little different approach:  
As I try to simulate regular retail trader behaviour, I think it would be also a good idea to treat the intraday data as an image, as most traders look at images and try to find patterns based on the visual indicators.  
Therefore the data preprocessing and the models in this repository are focused on Computer Vision techniques eather than Time Series models.

## Files and folders structure
The most important files are:  

**ETL**
* fetch_single_sublist_iqfeed.py - this module receives a list of tickers and a timeframe and returns the dataframes with the stock market data
* fetch_multithread_iqfeed.py - does the same as the file above, but utilizes multithreading
* std_filter_getter.py - scans which out of the stocks today pass the filters to be fed to our model. This is done to minimize the predicting time by narrowing the amount of data we want to feed and save us calculation time.

Part of the code in these files was deleted as the API for the software I use to get data (iqfeed) isn't open for a wide use.  
Any other API could be used instead.

**EDA**
* vol_std_calculate_statistics_v3.py - main module to analyze which way to filter stocks could benefit us in which way.
* save_vol_filtered_data_std_measured.py - a code to cache the files that meet centain volatility criteria, in order to save time loading data for the analysis.

**Backtesting**
* test_strategy_nn.py - simulates how a model would perform on the past data and how much profit and loss would it generate based on various parameters.

**Model training**:  
* trivial_nn_daytrade_v2_2.py
* trivial_nn_daytrade_v2_3.py  
* multitower_nn_daytrade_v2_4.py

These files contain different architectures of the model, so we can try them all and choose the best performing one

**Using the model in production**
* online_monitoring_30_mins_15_v_1.py - This code looks at the filtered stocks on the 30th minute, choose which one to buy and then sells them up to 15 minutes afterwads
* online_monitoring_v1_1_2.py - This code looks at the filtered stocks on the 100th minute, choose which one to buy and then sells them later
* online_monitoring_v1_2.py.py - This code looks at the filtered stocks on the 100th minute, choose which one to buy and then sells them later. Uses a different model to feed the stock data to and also has a different preprocessing technique

The latter 2 code files implement different preprocessing techniques, so I can try which one deals better with the data
