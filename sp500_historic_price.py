#!/usr/bin/env python
# coding: utf-8

# In[45]:


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

client =  MongoClient("mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/?retryWrites=true&w=majority")
db = client['mpt_optimizer']
collection = db['stock_returns_history']

sp500 = db.SP500_MarketCap
data=[]
data=sp500.find()


# In[46]:


enddate=dt.datetime.now()
startdte=enddate-dt.timedelta(days=1900)
print('Staring')  

for ticker in data:
    stockData = pdr.get_data_yahoo(ticker['Ticker'], startdte, enddate)
    stockData['ticker']=ticker['Ticker']
    stockData.reset_index(inplace=True)
    data_dict = stockData.to_dict("records")
    # Insert collection EOD prices
    print ("Inserting "+ticker['Ticker'])
    if (len(data_dict) == 0):
        print ('Skipping empty '+ ticker['Ticker'])
    else:
        collection.insert_many(data_dict)

print('Done')  

#from bson.json_util import dumps
##rsor = collection.find({})
#with open('collection.json', 'w') as file:
#    json.dump(json.loads(dumps(cursor)), file)




