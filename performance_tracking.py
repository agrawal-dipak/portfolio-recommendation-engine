# Initialising required libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Loading the dataset
data = pd.read_csv("D:/Users/dipak/workspace/temp/office project/portfolio-recommendation-engine/data/SNP 500 high low open close 5 years data.csv")
#The data is reversed, fixing it.
data=data.loc[::-1].reset_index(drop=True)

#Loading the apple data
stData = pd.read_csv("D:/Users/dipak/workspace/temp/office project/portfolio-recommendation-engine/data/AAPL.csv")

# X axis is price_date
price_date = pd.to_datetime(data['Date'])

# Y axis is price closing
price_close = data['Close/Last']

#apple prices
apple_close = stData['Close']
apple_date = pd.to_datetime(stData['Date'])

#Calculating percentages for both data
initialSNP = data.loc[0,'Close/Last']
initialStock = stData.loc[0,'Close']

for x in data.index:
    data.loc[x, 'Close/Last'] = data.loc[x, 'Close/Last']*100/initialSNP

for x in stData.index:
    stData.loc[x, 'Close'] = stData.loc[x, 'Close']*100/initialStock


# Plotting the timeseries graph of given dataset
plt.plot(price_date, price_close, label = 'SNP Performance')

plt.plot(apple_date, apple_close, label = 'Apple Performance')
plt.legend()

# Giving title to the graph
plt.title('Prices by Date')

# Giving x and y label to the graph
plt.xlabel('Price Date')
plt.ylabel('Price Close in percentage')


#plt.savefig('D:/Users/dipak/workspace/temp/office project/portfolio-recommendation-engine/performance.png')
plt.show()
