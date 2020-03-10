#!/usr/bin/env python
# coding: utf-8

# In[1]:


def random_portfolios(mean_returns, numAssets, cov_matrix, num_portfolios=1000, risk_free_rate=0.002):
    import numpy as np
    import pandas as pd
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(numAssets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    import numpy as np
    import pandas as pd
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def display_simulated_ef_with_random(data, numAssets, mean_returns, cov_matrix, num_portfolios=1000, risk_free_rate=0.002):
    import numpy as np
    import pandas as pd
    results, weights = random_portfolios(mean_returns, numAssets, cov_matrix, num_portfolios, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=data.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=data.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    return rp, sdp, rp_min, sdp_min, max_sharpe_allocation, min_vol_allocation, weights
