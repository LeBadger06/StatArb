import pandas as pd
import numpy as np
from scipy.stats import uniform

def stat_bootstrap(X, num_sim = 1000):
    q = 0.1
    T = len(X)
    X_star = []
    t = 1

    ## Case t = 1
    i_star = np.round(uniform(0, T))
    x_star = X.iloc[i_star]
    np.append(X_star, x_star)
    t += 1

    while t < T:
        U = uniform()
        if U < q: # X*(2) selected at random
            i_star = np.round(uniform(0, T))
            x_star = X.iloc[i_star]
        else: # X*(2) is next serial observation
            if i_star < T:
                x_star = X.iloc[i_star+1]
            else:
                x_star = X.iloc[0]
        np.append(X_star, x_star)
        t += 1
    df = pd.DataFrame(data = X_star, index = X.index)

