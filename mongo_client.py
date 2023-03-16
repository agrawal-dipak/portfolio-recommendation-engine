#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yfin
import scipy.optimize as sc
import plotly.graph_objects as go
import pandas as pd
import pandas as pd
from pymongo import MongoClient

yfin.pdr_override()

# insert S&P500 constituent data
sp_500_df= pd.read_csv('C:\sweta\portfolio-recommendation-engine\data\SP500_Companies.csv')
#stockList1=sp_500_df["Symbol"].astype(str).tolist()

#print(stockList1)
stockList= ['GOOGL','APPL','ACE', 'ABT', 'ANF', 'ACN', 'ADBE', 'AMD', 'AES', 'AET', 'AFL', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'ALXN']

enddate=dt.datetime.now()
startdte=enddate-dt.timedelta(days=365)
stockData = pdr.get_data_yahoo(stockList, startdte, enddate)

client =  MongoClient("mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/?retryWrites=true&w=majority")
#client=MongoClient('mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/journaldev?authSource=admin',
 #   server_selection_timeout=120000)
print("Connection done")
db = client['journaldev']
collection = db['stock_returns']
stockData.reset_index(inplace=True)
data_dict = stockData.to_dict("records")
# Insert collection EOD prices
collection.insert_many(data_dict)

#inset SP 500 constituents data
sp500_collection = db["sp500_data"]
sp_500_df.reset_index(inplace=True)
sp_data_dict = sp_500_df.to_dict("records") 
#print(sp_data_dict) 
# Insert SP500 data collection which is list of dictionary object
sp500_collection.insert_many(sp_data_dict)


# In[ ]:




