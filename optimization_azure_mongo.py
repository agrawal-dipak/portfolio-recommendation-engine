#!/usr/bin/env python
# coding: utf-8

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
import plotly.io as pio
from numpy.linalg import multi_dot



from slsqp_module import  *




client =  MongoClient("mongodb+srv://mongouser:market101@cluster0.fgigdzs.mongodb.net/?retryWrites=true&w=majority")
db = client['mpt_optimizer']

portfolio_coll = db['optimizer_summary']


storage_account_key = "qqf/6ZA1lakLtIA2UTDnf24HpnKWGqo8MHkFIrnLLMQORfCb3L0HkdIn7HFgqmRfi2YbiwmiEb/r+AStdIeCiw=="
storage_account_name = "storageopt1"
connection_string = "DefaultEndpointsProtocol=https;AccountName=storageopt1;AccountKey=qqf/6ZA1lakLtIA2UTDnf24HpnKWGqo8MHkFIrnLLMQORfCb3L0HkdIn7HFgqmRfi2YbiwmiEb/r+AStdIeCiw==;EndpointSuffix=core.windows.net"
container_name = "images"

FULLY_QUALIFIED_NAMESPACE = "market.servicebus.windows.net"
QUEUE_NAME = "portfolio_req_queue"

credential = DefaultAzureCredential()

servicebus_client = ServiceBusClient.from_connection_string(conn_str="Endpoint=sb://market.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=T4zNlsMxNEQw+5B923N2Ej1JUZ91ITkz4+ASbCHOjws=")

### Prepare the data frame for process and return mean/var matrix
def prepare_data(msg,Optver):
    sp500 = db['SP500_MarketCap']
    tickers = pd.DataFrame(list(sp500.find()))

    ticker_list=tickers['Ticker'].tolist()
    start = dt.datetime(2022, 3, 14, 7, 51, 4)
    end = dt.datetime(2022, 9, 30, 7, 52, 4)
    collection = db['stock_returns_history']
    data = pd.DataFrame(list(list (collection.find( {"Date":{"$gt": start , "$lt":end},'ticker':{"$in":ticker_list}}))))
    pivoted = data.pivot(index='Date', columns='ticker', values='Close').reset_index()
    pivoted.columns.name=None
    df_ii = pivoted.set_index(pd.DatetimeIndex(pivoted["Date"])).drop("Date",axis=1)
    returns=df_ii.pct_change()
    ## Dataset ready . retruns have the daily return matrix of filtered stock.

    #get top 20 stock list
    if (Optver=="MC"):
        top_20_stocks_mean = (returns.mean()*252).sort_values(ascending=False).head(20)
    else:
        top_20_stocks_mean = (returns.mean()).sort_values(ascending=False).head(20)
    tkr = (top_20_stocks_mean.to_dict().keys())
    filtered = returns[tkr]
    #for slsqp do not multiply with 252
    cov_matrix=filtered.cov()
    return top_20_stocks_mean,cov_matrix,filtered


# function to perform upload
def uploadToBlobStorage(file_path,file_name):
   blob_service_client = BlobServiceClient.from_connection_string(connection_string)
   blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
   with open(file_path,"rb") as data:
      blob_client.upload_blob(data)
      print("Uploaded " +file_name)

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
                     
                     optimize_mc(str(msg)+"mc")
                     optimize_slsqp(str(msg)+"slsqa")
        # Close credential when no longer needed.
         await credential.close()


def optimize_mc(msg):

        opt_port_returns=[]
        opt_port_vol=[]
        opt_port_weights=[]
        sharpe_ratios=[]
        num_portfolios = 5000

        indvidual_mean_ret,cv,tickers =  prepare_data(msg,"MC")
        var_matrix = cv*252
        numOfAsset = len(indvidual_mean_ret.index)

        for port in range(num_portfolios):
        
            weights=np.random.random(numOfAsset)
            weights=weights/np.sum(weights)

            opt_port_weights.append(weights)
            returns_sample=np.dot(weights,indvidual_mean_ret)
            opt_port_returns.append(returns_sample)

            var=var_matrix.mul(weights,axis=0).mul(weights,axis=1).sum().sum()
            sd=np.sqrt(var)
            #ann_sd=sd*np.sqrt(250)
            ann_sd=sd
            opt_port_vol.append(ann_sd)   
            sharpe_ratios.append(returns_sample/ann_sd)


        data1={'Returns':opt_port_returns, 'Volatility':opt_port_vol,'SharpRatio':sharpe_ratios}
        for counter, symbol in enumerate (tickers):
            data1[symbol] = [w[counter] for w in opt_port_weights ]

        portfolios_V1=pd.DataFrame.from_dict(data1, orient='index')

        abc=portfolios_V1.transpose()

        min_vol_port=abc.iloc[abc['Volatility'].idxmin()]
        max_sharp_ratio_portfolio=abc.iloc[abc['SharpRatio'].idxmax()]
        max_return_port=abc.iloc[abc['Returns'].idxmax()]


        filename=str(msg)+".png"
        

        rf=0.01
        # optimal_risk_port=abc.iloc[((abc['Returns']-rf)/abc['Volatility']).idxmax()]

        min_vol_port=abc.iloc[abc['Volatility'].idxmin()]
        max_sharp_ratio_portfolio=abc.iloc[abc['SharpRatio'].idxmax()]
        max_return_port=abc.iloc[abc['Returns'].idxmax()]
        mydict = { "runid": str(msg), "min_vol_port": min_vol_port.to_dict(),"max_risk_port":max_return_port.to_dict() ,"max_sharp_ratio":max_sharp_ratio_portfolio.to_dict(),"image" :filename}
        
        x = portfolio_coll.insert_one(mydict)
        plt.subplots(figsize=(8,8))
        plt.scatter(abc['Volatility'],abc['Returns'],c=sharpe_ratios)
        
        clb=plt.colorbar()
        clb.ax.set_title("Sharpe Ratio")
        plt.scatter(min_vol_port[1],min_vol_port[0],color='b',marker='*',s=500)
        plt.scatter(max_sharp_ratio_portfolio[1],max_sharp_ratio_portfolio[0],color='y',marker='*',s=500)
        
        plt.grid()
        plt.xlabel("Risk (Volatility)")
        plt.ylabel("Expected Returns")
        plt.savefig(filename)
        uploadToBlobStorage(filename,filename)

        #plotly code to embedd data
        #fig= px.scatter(df_chart, x="Volatility", y="Returns", color="SharpRatio" ,hover_data=['SharpRatio','AAPL weight','GOOGL weight','MSFT weight','NKE weight'])
        ##fig.show()
        #fig.write_html("plotly.html")
        #optimize("msg")

def optimize_slsqp(msg):
    
    meanReturns,covMatrix,tickers=prepare_data(msg,"slsqp")
    numOfAsset = len(meanReturns.index)
    wt=(100/numOfAsset)/100
    weights=np.empty(numOfAsset); weights[:]=wt
    returns,std=portfolioPerformance(weights,meanReturns,covMatrix)
    
    filename=str(msg)+".png"
    dict_results,fig=EF_graph(meanReturns, covMatrix,msg,filename)
    x = portfolio_coll.insert_one(dict_results)
    fig.savefig(filename)
    uploadToBlobStorage(filename,filename)

asyncio.run(run())
print("Done getting messages")
print("-----------------------")

