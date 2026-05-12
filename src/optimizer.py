import numpy as np

def optimize_portfolio(returns, rf=0.01):

    mean_returns = returns.mean()
    cov = returns.cov()

    n = len(mean_returns)
    best_sharpe = -999
    best_weights = None

    for _ in range(5000):
        weights = np.random.random(n)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

        sharpe = (portfolio_return - rf) / portfolio_vol

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

    return best_weights, best_sharpe
