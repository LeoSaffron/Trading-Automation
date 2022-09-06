# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:26:48 2020

@author: jonsnow
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 03:08:59 2020

@author: jonsnow
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 02:15:11 2020

@author: jonsnow
"""

# iqfeed.py

import sys
import pandas as pd
import os
from datetime import datetime
from datetime import date


def main(ticker_list, day_start, time_start, time_end, verbose, error_verbose):
    faulty_tickers = []
    df_list = []
    return df_list, faulty_tickers