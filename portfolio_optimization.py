# Import base libraries
from numpy import *
from numpy.linalg import multi_dot
import pandas as pd
# import yfinance as yf
import plotly
import plotly.express as px
import warnings
import cufflinks as cf
import scipy.optimize as sco

cf.set_config_file(offline=True, dimensions=(1000, 600))
setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)


# Maximizing sharpe ratio
def max_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]


# Minimize the variance
def min_variance(weights):
    return portfolio_stats(weights)[1] ** 2


def min_volatility(weights):
    return portfolio_stats(weights)[1]


# Define portfolio stats function
def portfolio_stats(weights):
    weights = array(weights)[:, newaxis]
    port_rets = weights.T @ array(returns.mean() * 252)[:, newaxis]
    port_vols = sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))

    return array([port_rets, port_vols, port_rets / port_vols]).flatten()


def portfolio_simulation(returns):
    # Initialize the lists
    rets = []
    vols = []
    wts = []

    # Simulate 5,000 portfolios
    for i in range(numOfPortfolio):
        # Generate random weights
        weights = random.random(numOfAsset)[:, newaxis]

        # Set weights such that sum of weights equals 1
        weights /= sum(weights)

        # Portfolio statistics
        rets.append(weights.T @ array(returns.mean() * 252)[:, newaxis])
        vols.append(sqrt(multi_dot([weights.T, returns.cov() * 252, weights])))
        wts.append(weights.flatten())

    # Create a dataframe for analysis
    portdf = 100 * pd.DataFrame({
        'port_rets': array(rets).flatten(),
        'port_vols': array(vols).flatten(),
        'weights': list(array(wts))
    })

    portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']

    return round(portdf, 2)


cf.set_config_file(offline=True, dimensions=(1000, 600))

px.defaults.template, px.defaults.width, px.defaults.height = "plotly_white", 1000, 600

# Ignore warnings

warnings.filterwarnings('ignore')

# Nasdaq-listed stocklist
symbols = ['AAPL', 'AMZN', 'MSFT', 'NVDA', 'TSLA']

# Number of assets
numOfAsset = len(symbols)

# Number of portfolio for optimization
numOfPortfolio = 5000

# Fetch data from yahoo finance for last six years
# nasdaqStocks = yf.download(symbols, start='2017-01-01', end='2023-03-15', progress=False)['Adj Close']
# nasdaqStocks.to_csv('data/symbolData.csv')

# Load locally stored data
df = pd.read_csv('data/symbolData.csv', index_col=0, parse_dates=True)


# Calculate returns
returns = df.pct_change().fillna(0)
# print(returns)

# Create a dataframe for analysis
temp = portfolio_simulation(returns)
# print(temp)

# Plot annualized return and volatility
# d.DataFrame({
#   'Annualized Return': round(returns.mean()*252*100,2),
#   'Annualized Volatility': round(returns.std()*sqrt(252)*100,2)
# ).iplot(kind='bar', shared_xaxes=True, subplots=True)

# Get the max sharpe portfolio stats
# temp.iloc[temp.sharpe_ratio.idxmax()]

# Max sharpe ratio portfolio weights
msrpwts = temp['weights'][temp['sharpe_ratio'].idxmax()]

# Allocation to achieve max sharpe ratio portfolio
dictionary = dict(zip(symbols, msrpwts))
print(dictionary)

# Define initial weights
initial_wts = numOfAsset * [1. / numOfAsset]
bnds = tuple((0, 1) for x in range(numOfAsset))
cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

# Optimizing for maximum sharpe ratio
opt_sharpe = sco.minimize(max_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Optimizing for minimum variance
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Efficient frontier params
targetrets = linspace(0.30, 0.60, 100)
tvols = []
# Each asset boundary ranges from 0 to 1 bounds
bnds = tuple((0, 1) for x in range(numOfAsset))

for tr in targetrets:
    ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
               {'type': 'eq', 'fun': lambda x: sum(x) - 1})

    opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)

    tvols.append(opt_ef['fun'])

targetvols = array(tvols)
# Dataframe for EF
efport = pd.DataFrame({
    'targetrets': around(100 * targetrets[14:], 2),
    'targetvols': around(100 * targetvols[14:], 2),
    'targetsharpe': around(targetrets[14:] / targetvols[14:], 2)
})

# Plot efficient frontier portfolio
fig = px.scatter(
    efport, x='targetvols', y='targetrets', color='targetsharpe',
    labels={'targetrets': 'Expected Return', 'targetvols': 'Expected Volatility', 'targetsharpe': 'Sharpe Ratio'},
    title="Efficient Frontier Portfolio"
).update_traces(mode='markers', marker=dict(symbol='cross'))

# Plot maximum sharpe portfolio
fig.add_scatter(
    mode='markers',
    x=[100 * portfolio_stats(opt_sharpe['x'])[1]],
    y=[100 * portfolio_stats(opt_sharpe['x'])[0]],
    marker=dict(color='red', size=20, symbol='star'),
    name='Max Sharpe'
).update(layout_showlegend=False)

# Plot minimum variance portfolio
fig.add_scatter(
    mode='markers',
    x=[100 * portfolio_stats(opt_var['x'])[1]],
    y=[100 * portfolio_stats(opt_var['x'])[0]],
    marker=dict(color='green', size=20, symbol='star'),
    name='Min Variance'
).update(layout_showlegend=False)

# Show spikes
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()
