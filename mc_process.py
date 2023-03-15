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


client =  MongoClient("mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/?retryWrites=true&w=majority")
db = client['mpt_optimizer']
collection = db['stock_daily_price']

storage_account_key = ""
storage_account_name = "storageopt1"
connection_string = ""
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
                     optimize()
        # Close credential when no longer needed.
         await credential.close()


def optimize():
    test = db.stock_daily_price
    #convert entire collection to Pandas dataframe
    data = pd.DataFrame(list(test.find()))

    # In[9]:

    pivoted = data.pivot(index='Date', columns='ticker', values='Close').reset_index()
    pivoted.columns.name=None


    df_ii = pivoted.set_index(pd.DatetimeIndex(pivoted["Date"])).drop("Date",axis=1)

    returns=df_ii.pct_change()

    weights=[0.2,0.4,0.3,0.1]
    port_ret=returns.dot(weights)
    var_matrix=returns.cov()*252

    portfolio_variance=np.transpose(weights)@var_matrix@weights
    portfolio_vol=np.sqrt(portfolio_variance)
    print ("Variance ",portfolio_variance )
    print ("Risk ",portfolio_vol )


    # In[17]:


    num_portfolios=10000
    indvidual_ret=df_ii.resample('Y').last().pct_change().mean()
    opt_port_returns=[]
    opt_port_vol=[]
    opt_port_weights=[]
    for port in range(num_portfolios):
        weights=np.random.random(4)
        weights=weights/np.sum(weights)
        opt_port_weights.append(weights)
        returns_t=np.dot(weights,indvidual_ret)
        opt_port_returns.append(returns_t)
        
        var=var_matrix.mul(weights,axis=0).mul(weights,axis=1).sum().sum()
        
        sd=np.sqrt(var)
        ann_sd=sd*np.sqrt(250)
        opt_port_vol.append(ann_sd)   
    data1={'Returns':opt_port_returns, 'Volatility':opt_port_vol}
    for counter, symbol in enumerate (df_ii.columns.tolist()):
        data1[symbol+' weight'] = [w[counter] for w in opt_port_weights ]
        
    portfolios_V1=pd.DataFrame.from_dict(data1, orient='index')

    abc=portfolios_V1.transpose()


    # In[21]:

    # In[38]:

    pid = os.getpid()
    filename=str(pid)+".png"

    portfolio_coll = db['optimizer_summary']

    min_vol_port=abc.iloc[abc['Volatility'].idxmin()]

    rf=0.01
    optimal_risk_port=abc.iloc[((abc['Returns']-rf)/abc['Volatility']).idxmax()]

    mydict = { "runid": 1, "min_vol_port": min_vol_port.to_dict(),"max_risk_port":optimal_risk_port.to_dict() ,"image" :filename}

    x = portfolio_coll.insert_one(mydict)


    # In[29]:


    plt.subplots(figsize=(8,8))
    plt.scatter(abc['Volatility'],abc['Returns'],marker='o',color='g',s=15,alpha=0.3)
    plt.scatter(min_vol_port[1],min_vol_port[0],color='r',marker='*',s=500)
    plt.scatter(optimal_risk_port[1],optimal_risk_port[0],color='b',marker='*',s=500)
    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Returns")
    plt.savefig(filename)


    # In[39]:

    uploadToBlobStorage(filename,filename)

asyncio.run(run())
print("Done getting messages")
print("-----------------------")
