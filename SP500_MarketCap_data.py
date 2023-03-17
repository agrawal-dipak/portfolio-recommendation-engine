#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yfin
import scipy.optimize as sc
import plotly.graph_objects as go
import pandas as pd
import pandas as pd
from pymongo import MongoClient

# insert S&P500 constituent data
sp_500_df= pd.read_csv('C:\\Users\\ratheesh\\azure\\portfolio-recommendation-engine\\data\\SP500_MarketCap_data.csv')


client =  MongoClient("mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/?retryWrites=true&w=majority")

db = client['mpt_optimizer']
collection = db['SP500_MarketCap']
sp_500_df.reset_index(inplace=True)
sp_data_dict = sp_500_df.to_dict("records") 
collection.insert_many(sp_data_dict)


# In[ ]:




