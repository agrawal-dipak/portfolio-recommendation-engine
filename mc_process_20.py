#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import scipy.optimize as sc
import plotly.graph_objects as go
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
import os
import asyncio
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
from azure.identity.aio import DefaultAzureCredential
import random



client =  MongoClient("mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/?retryWrites=true&w=majority")
db = client['mpt_optimizer']
#collection = db['stock_daily_price']





storage_account_key = "qqf/6ZA1lakLtIA2UTDnf24HpnKWGqo8MHkFIrnLLMQORfCb3L0HkdIn7HFgqmRfi2YbiwmiEb/r+AStdIeCiw=="
storage_account_name = "storageopt1"
connection_string = "DefaultEndpointsProtocol=https;AccountName=storageopt1;AccountKey=qqf/6ZA1lakLtIA2UTDnf24HpnKWGqo8MHkFIrnLLMQORfCb3L0HkdIn7HFgqmRfi2YbiwmiEb/r+AStdIeCiw==;EndpointSuffix=core.windows.net"
container_name = "images"

FULLY_QUALIFIED_NAMESPACE = "market.servicebus.windows.net"
QUEUE_NAME = "portfolio_req_queue"

credential = DefaultAzureCredential()

servicebus_client = ServiceBusClient.from_connection_string(conn_str="Endpoint=sb://market.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=T4zNlsMxNEQw+5B923N2Ej1JUZ91ITkz4+ASbCHOjws=")

# function to perform upload
def uploadToBlobStorage(file_path,file_name):
   blob_service_client = BlobServiceClient.from_connection_string(connection_string)
   blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
   with open(file_path,"rb") as data:
      blob_client.upload_blob(data)
      print("Uploaded {file_name}.")

async def run():
    # create a Service Bus client using the credential
    async with servicebus_client:
         # get a Queue Sender object to send messages to the queue
         # get the Queue Receiver object for the queue
         receiver = servicebus_client.get_queue_receiver(queue_name=QUEUE_NAME)
         async with receiver:
                received_msgs = await receiver.receive_messages(max_wait_time=5, max_message_count=1)
                for msg in received_msgs:
                     print("Received: " + str(msg))
                     # complete the message so that the message is removed from the queue
                     await receiver.complete_message(msg)
                     optimize(msg)
        # Close credential when no longer needed.
         await credential.close()


def optimize(msg):
    #test = db.stock_daily_price
    #convert entire collection to Pandas dataframe
    sp500 = db['SP500_MarketCap']
    tickers = pd.DataFrame(list(sp500.find()))
    sample = tickers.sample(n=20)
    ticker_list=sample['Ticker'].tolist()
    #for testing
    #ticker_list=['GOOGL','MSFT','NKE','TSLA'] 
    start = dt.datetime(2022, 3, 14, 7, 51, 4)
    end = dt.datetime(2022, 9, 30, 7, 52, 4)

    collection = db['stock_returns_history']

    data = pd.DataFrame(list(list (collection.find( {"Date":{"$gt": start , "$lt":end},'ticker':{"$in":ticker_list}}))))
    
    # In[9]:

    pivoted = data.pivot(index='Date', columns='ticker', values='Close').reset_index()
    pivoted.columns.name=None


    df_ii = pivoted.set_index(pd.DatetimeIndex(pivoted["Date"])).drop("Date",axis=1)

    returns=df_ii.pct_change()

    returns.to_csv('file1.csv')
    print(returns)
    num_portfolios=5000

    #indvidual_mean_ret=df_ii.resample('Y').last().pct_change().mean()
   # portfolio intial paramters
    indvidual_mean_ret= returns.mean()*252

    print (returns)
    var_matrix=returns.cov()*252
    print ('co-var')
    print(var_matrix)


    opt_port_returns=[]
    opt_port_vol=[]
    opt_port_weights=[]
    sharpe_ratios=[]

    numOfAsset = len(indvidual_mean_ret.index)
    print (numOfAsset)

    for port in range(num_portfolios):
        weights=np.random.random(numOfAsset)
        weights=weights/np.sum(weights)
        
        opt_port_weights.append(weights)
        #returns_t=np.dot(weights,indvidual_mean_ret)
        returns_sample=np.dot(weights,indvidual_mean_ret)
        opt_port_returns.append(returns_sample)
                
        var=var_matrix.mul(weights,axis=0).mul(weights,axis=1).sum().sum()
        
        sd=np.sqrt(var)
        ann_sd=sd*np.sqrt(250)
        opt_port_vol.append(ann_sd)   
        sharpe_ratios.append(returns_sample/ann_sd)
       

    data1={'Returns':opt_port_returns, 'Volatility':opt_port_vol}
    for counter, symbol in enumerate (df_ii.columns.tolist()):
        data1[symbol] = [w[counter] for w in opt_port_weights ]
        
    portfolios_V1=pd.DataFrame.from_dict(data1, orient='index')

    abc=portfolios_V1.transpose()

    pid = random.randint(1000,2000)
    filename=str(msg)+str(pid)+".png"

    print (filename)

    portfolio_coll = db['optimizer_summary']


    rf=0.01
   # optimal_risk_port=abc.iloc[((abc['Returns']-rf)/abc['Volatility']).idxmax()]
    
    min_vol_port=abc.iloc[abc['Volatility'].idxmin()]
    #max_sharp_ratio_portfolio=abc.iloc[abc['SharpRatio'].idxmax()]
    max_return_port=abc.iloc[abc['Returns'].idxmax()]

    mydict = { "runid": pid, "min_vol_port": min_vol_port.to_dict(),"max_risk_port":max_return_port.to_dict() ,"image" :filename}

    x = portfolio_coll.insert_one(mydict)

    plt.subplots(figsize=(8,8))
    plt.scatter(abc['Volatility'],abc['Returns'],c=sharpe_ratios)
    plt.scatter(min_vol_port[1],min_vol_port[0],color='b',marker='*',s=500)
    
    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Returns")
    plt.savefig(filename)

    uploadToBlobStorage(filename,filename)
    print(min_vol_port)
    print(max_return_port)
    #print(max_sharp_ratio_portfolio)

#plotly code to embedd data
#fig= px.scatter(df_chart, x="Volatility", y="Returns", color="SharpRatio" ,hover_data=['SharpRatio','AAPL weight','GOOGL weight','MSFT weight','NKE weight'])
##fig.show()
#fig.write_html("plotly.html")

#asyncio.run(run())
optimize("test")
print("Done getting messages")
print("-----------------------")

