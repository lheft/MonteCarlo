#!/usr/bin/env python
# coding: utf-8

# In[91]:


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:31:32 2022
 
@author: ktjonasam
"""
 
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from itertools import chain, repeat, count, islice
import itertools
from tqdm import tqdm
import statistics

#import os
#os.chdir(r"X:\Sales\Maier Models\Py Tools\MC_Scott")
 


# In[58]:


endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=2180)
 
# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData=yf.download(stocks,start,end,threads=True)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix
 
import_stock_list_raw=pd.read_excel('/Users/loganheft/Desktop/Thematic.xlsx',parse_dates=True)
 
 
#Stocklist filtered for ETF that have a 5 year track list
stock_list_filtered=import_stock_list_raw[(import_stock_list_raw['ETF Inception Date']<'01-01-2018')]
 
stock_list_filtered
 


# # 5yr Single Point Portfolio (No replacement)

# In[96]:


#Generate portfolio Reutrns using single point 5yr (Annualized Total Returns)
#n_portfolios=10000
n_portfolios=5
set_port_returns=pd.DataFrame()
portfolio_dict={}
for i in range(n_portfolios):
    # time.sleep(1)
    stocklist=list(np.random.choice(stock_list_filtered['Ticker'],4))
    print(stocklist)
    
    portfolio_name="Portfolio {0}".format(i)
    portfolio_dict[portfolio_name] = stocklist
    temp_select_tickers=stock_list_filtered[["Ticker","5YR Return"]].loc[stock_list_filtered["Ticker"].isin(stocklist)]
    temp_select_tickers['Weights']=.25
    
    portfolio_returns=np.cumprod(np.inner(temp_select_tickers["5YR Return"],temp_select_tickers['Weights']))
    portfolio_returns_str=(str(portfolio_returns).lstrip('[').rstrip(']'))
    print(f'{portfolio_returns_str}%') 
    
    temp=pd.DataFrame([portfolio_name,float(portfolio_returns)]).T
    set_port_returns=pd.concat([set_port_returns,temp],axis=0)
    print('')
    

set_port_returns.columns=(["Port_Name",'Returns'])
set_port_returns=set_port_returns.sort_values(by=['Returns'],ascending=False)
print(set_port_returns)
print('')
hit_Rate=(len(set_port_returns[set_port_returns['Returns']>0])/n_portfolios)*100
port_single_point_mean=set_port_returns['Returns'].mean().round(3)
median=round(statistics.median(set_port_returns['Returns']),3)
minimum=round(min(set_port_returns['Returns']),3)
maximum=round(max((set_port_returns['Returns'])),3)
port_std_dev=np.std(set_port_returns['Returns']).round(3)

print(f'{hit_Rate}% of portfolios return positive results')
print(f'The average return of the randomly generated portfolios was {port_single_point_mean}%')  
print(f'The median return of the randomly generated portfolios was {median}%')
print(f'The portfolio with the lowest return was {minimum}%')
print(f'The portfolio with the highest return was {maximum}%')
print(f'The standard deviation of all the portfolios is {port_std_dev}%')
 
 


# In[97]:


sns.displot(set_port_returns['Returns'],kde=True)
 
# Displaying the averages --- You can positions this at the end of the script
print("Average Portfolio Hit Rate is " + str(hit_Rate*100) +"% After " +str(n_portfolios) +" Portfolio Simulations")
print("Average Portfolio Returns is " + str(port_single_point_mean.round(3)) +"% After " +str(n_portfolios) +" Portfolio Simulations")
 
######## MC Simulation utilizing Daily Returns ######


# In[8]:


#### Generate the Data for the MC Daily returns
stock_list_mc=list(stock_list_filtered['Ticker']) #### This reads from the set or portoflio iterations
stockData=yf.download(stock_list_mc,startDate,endDate,threads=True)
stockData = stockData['Close']


# #  MonteCarlo Simulation 

# In[16]:


# Define the risk-free rate
risk_free_rate = 0.03

# Create an empty DataFrame to store the portfolio names and Sharpe ratios
portfolio_ratios = pd.DataFrame(columns=['Portfolio', 'Sharpe Ratio','Average Return','Standard Deviation','Range of Returns','Median Return'])

for portfolio_name, portoflio_holdings in portfolio_dict.items():
    try:
        print (portfolio_name)
        stock_list=portfolio_dict[portfolio_name]
        returns = stockData[stock_list].pct_change()
        
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        corrMatrix=returns.corr()
        
        # handling of NaN values
        covMatrix=np.nan_to_num(covMatrix)
        meanReturns = np.nan_to_num(meanReturns)
       
        weights = np.array([.25,.25,.25,.25])
        weights /= np.sum(weights)
       
        # Monte Carlo Method
        T = 365*5 #timeframe in days - Simulates 5 Daily Year Returns (Current Run time is 10it/s)
        
        meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
        meanM = meanM.T
       
        portfolio_sims = np.full(shape=(T, n_portfolios), fill_value=0.0) # This sets up a blank array for the returns for each port iterations
       
        initialPortfolio = 10000
       
        # for m in range(0, mc_sims):
        Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
        L = np.linalg.cholesky(covMatrix)

        dailyReturns = meanM + np.inner(L, Z)
        Port_MC_Sim=portfolio_sims[:,n_portfolios-1] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio
        
        # Find the average return
        average_return = np.mean(Port_MC_Sim)
        
        # Find the standard deviation
        std_return = np.std(Port_MC_Sim)
        
        # Find the range of returns
        range_return = (np.max(Port_MC_Sim) - np.min(Port_MC_Sim))
        
        # Find the median return
        median_return = np.median(Port_MC_Sim)
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (average_return - risk_free_rate) / std_return
        
        # Append the portfolio name and ratios to the DataFrame
        portfolio_ratios = portfolio_ratios.append({'Portfolio': portfolio_name, 'Sharpe Ratio': sharpe_ratio, 'Average Return':average_return, 'Standard Deviation':std_return, 'Range of Returns':range_return, 'Median Return':median_return}, ignore_index=True)
        temp_df_for_plot = pd.DataFrame(Port_MC_Sim,columns=[portfolio_name])
        results_for_plots = pd.concat([results_for_plots, temp_df_for_plot],axis=1)
        print(results_for_plots)
    
    except:
        pass
        ### This is to pass for errors within the cov matrix (Exception Raised: Matrix is not positive definite)

plt.plot(results_for_plots)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()   

# Print the resulting DataFrame
print(portfolio_ratios)


# # Simulation replacing the worst performer in a portfolio at the end of each year

# In[78]:


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n_portfolios = 10
portfolio_returns = {}
tickers=list(stock_list_filtered['Ticker'])

def get_return_for_ticker_in_year(ticker, year):
    # Get stock data for the ticker from Yahoo Finance
    data = yf.download(ticker, start='2018-01-01', end='2022-12-31')

    # Calculate the return for the year
    return_val = (data['Close'][-1] - data['Close'][0]) / data['Close'][0]

    return return_val

# Initialize a list to store the values of each portfolio over time
portfolio_values_over_time = [[] for _ in range(n_portfolios)]
sum_returns = [0]*5

for i in range(n_portfolios):
    stocklist = list(np.random.choice(tickers, 4))
    portfolio_name = "Portfolio {0}".format(i)
    portfolio_returns[portfolio_name] = {}
    portfolio_value = 10000 # reset the value of the portfolio at the beginning of each iteration 
    portfolio_values_over_time[i].append(portfolio_value)
    for year in range(5):
        returns = []
        for ticker in stocklist:
            # Retrieve the return for each ticker for the current year
            return_val = get_return_for_ticker_in_year(ticker, year)
            returns.append(return_val)
            sum_returns[year] += return_val

        portfolio_returns[portfolio_name][year] = returns
        # Update the portfolio value based on the returns of each stock, giving them equal weighting
        portfolio_value = portfolio_value * (1 + sum(returns) / len(returns))
        portfolio_values_over_time[i].append(portfolio_value)

        # Remove the worst performing ticker
        worst_ticker_index = returns.index(min(returns))
        print(f'{stocklist[worst_ticker_index]} is being deleted')

        stocklist.pop(worst_ticker_index)
        
        # Add a new ticker to the portfolio
        new_ticker = random.choice(tickers)
        stocklist.append(new_ticker)
        tickers.remove(new_ticker)
        print(stocklist)
    print('')
    print('This is a new portfolio')

# Plot the value of the portfolios over time
for i in range(n_portfolios):
    plt.plot(portfolio_values_over_time[i], label=f'Portfolio {i}')
plt.xlabel('Year')
plt.ylabel('Portfolio Value')
plt.title('Value of Portfolio over Time')
plt.legend()
plt.show()

# calculate the average return
average_returns = [returns/n_portfolios for returns in sum_returns]
# Plot the average returns over time
plt.plot(average_returns)
plt.xlabel('Year')
plt.ylabel('Average Returns')
plt.title('Average Returns over Time')
plt.show()


# # Simulation removing and replacing the top performer in each portfolio

# In[79]:


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n_portfolios = 10
portfolio_returns = {}
tickers=list(stock_list_filtered['Ticker'])

def get_return_for_ticker_in_year(ticker, year):
    # Get stock data for the ticker from Yahoo Finance
    data = yf.download(ticker, start='2018-01-01', end='2022-12-31')

    # Calculate the return for the year
    return_val = (data['Close'][-1] - data['Close'][0]) / data['Close'][0]

    return return_val

# Initialize a list to store the values of each portfolio over time
portfolio_values_over_time = [[] for _ in range(n_portfolios)]
sum_returns = [0]*5

for i in range(n_portfolios):
    stocklist = list(np.random.choice(tickers, 4))
    portfolio_name = "Portfolio {0}".format(i)
    portfolio_returns[portfolio_name] = {}
    portfolio_value = 10000 # reset the value of the portfolio at the beginning of each iteration 
    portfolio_values_over_time[i].append(portfolio_value)
    for year in range(5):
        returns = []
        for ticker in stocklist:
            # Retrieve the return for each ticker for the current year
            return_val = get_return_for_ticker_in_year(ticker, year)
            returns.append(return_val)
            sum_returns[year] += return_val

        portfolio_returns[portfolio_name][year] = returns
        # Update the portfolio value based on the returns of each stock, giving them equal weighting
        portfolio_value = portfolio_value * (1 + sum(returns) / len(returns))
        portfolio_values_over_time[i].append(portfolio_value)

        # Remove the worst performing ticker
        best_ticker_index = returns.index(max(returns))
        print(f'{stocklist[best_ticker_index]} is being deleted')

        stocklist.pop(best_ticker_index)
        
        # Add a new ticker to the portfolio
        new_ticker = random.choice(tickers)
        stocklist.append(new_ticker)
        tickers.remove(new_ticker)
        print(stocklist)
    print('')
    print('This is a new portfolio')

# Plot the value of the portfolios over time
for i in range(n_portfolios):
    plt.plot(portfolio_values_over_time[i], label=f'Portfolio {i}')
plt.xlabel('Year')
plt.ylabel('Portfolio Value')
plt.title('Value of Portfolio over Time')
plt.legend()
plt.show()

# calculate the average return
average_returns = [returns/n_portfolios for returns in sum_returns]
# Plot the average returns over time
plt.plot(average_returns)
plt.xlabel('Year')
plt.ylabel('Average Returns')
plt.title('Average Returns over Time')
plt.show()


# In[ ]:




