import numpy as np

def compute_returns(data):
    return data.pct_change().dropna()

def compute_covariance(returns):
    return returns.cov()
