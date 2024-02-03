"""This module was used to download the initial data set"""

import pandas as pd
import yfinance as yf

etf_list = ['SPY', 'QQQ', 'SOXX', 'VOO', 'IWM',
            'SQQQ', 'SOXS', 'XLE', 'XLK', 'XLV', 'IVV',
            'INDA', 'ARKK', 'EEM', 'FXI',
            'SXRT',
            'TLT', 'LQD', 'IEF', 'SHY',
            'VIXY']

hist = yf.download( etf_list, period="1d", start="2010-01-01", end="2024-01-21" )
hist.to_csv( r'/home/victor/Documents/WBS Algo Trading/yahoo_data.csv', 
             sep = ';',
             index_label = ['Date'])

hist_read= pd.read_csv( r'/home/victor/Documents/WBS Algo Trading/yahoo_data.csv', 
                   sep = ';',
                   parse_dates = True,
                   header=[0,1,2])
hist_read.columns = hist_read.columns.droplevel(2)
hist_read = hist_read.set_index( hist_read.iloc[:,0] )
hist_read.index.name = 'timestamp'
hist_read = hist_read.drop( columns=hist_read.columns[0] )
