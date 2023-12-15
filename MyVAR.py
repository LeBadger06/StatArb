import numpy as np
import MyOLS as MyOLS
from numpy import linalg
import pandas as pd

def MyVAR(ret, lags, print_debug = False):

    # Extract column names
    features = ret.columns

    ret2 = ret.copy()

    # --------------------------------------
    # Make matrix of explanatory variables
    # --------------------------------------
    # Loop through each lags
    for i in range(1, lags + 1):
        # Loop through each features
        for j in features:
            # Add lag i of feature j to the dataframe
            ret2[f"{j}_Lag_{i}"] = ret[j].shift(i)
    ret2.dropna(inplace=True)
    X = ret2.drop(columns=features)

    # Insert an intercept column
    X.insert(0, "Intercept", 1)

    # --------------------------------------
    # Make matrix of dependent variables
    # --------------------------------------
    Y = ret2[features]

    # Store results
    Results = dict()
    Results['X'] = X
    Results['Y'] = Y

    # Do linear regression
    mymodel = MyOLS.MyOLS(Y, X)
    if print_debug: print("X.shape[0], X.shape[1] ", X.shape[0], X.shape[1])
    LR_results = mymodel.fit()

    # For stability check, need to write the companion form and check eigenvalues
    Psi = pd.DataFrame(LR_results['B_hat'].T)
    # drop first column
    Psi = Psi.iloc[:, 1:]
    # reset column numbers
    Psi.columns = range(Psi.columns.size)

    dim = len(features) * (lags - 1)
    Identity = np.eye(dim)
    Zeroes = np.zeros((dim, len(features)))
    IdentityZeroes = np.concatenate([Identity, Zeroes], axis=1)
    Psi = pd.concat([Psi, pd.DataFrame(IdentityZeroes)], axis=0)
    # print(Psi.head(len(Psi)))

    # Get eigenvalues
    try:
        eig = linalg.eigvals(Psi)
    except np.linalg.LinAlgError:
        print("Can't compute eigenvalues.")

    # Stable if absolute value less then 1
    # print(pd.DataFrame(eig))
    # print(np.abs(pd.DataFrame(eig)))
    isStable = False
    isStable = (np.abs(pd.DataFrame(eig)) < 1).eq(True).all()
    # print(isStable)

    # Store results
    Results['LR'] = LR_results
    Results['VAR_stable'] = isStable

    return Results


def Granger_likelihood_ratio_test(returns, lag, alpha=0.05, print_res=False):
    # Works with pairs only for now
    # We should improve this by extending this for more than 2

    import MyVAR
    import numpy as np

    # Unrestricted model
    VARres = MyVAR.MyVAR(returns.astype(float), lag)
    if print_res: print(VARres['LR']['df'].head(3 * lag))
    SSE_U = np.diagonal(VARres['LR']['SS_res'])
    if print_res: print("SSE_U ", SSE_U)
    n = VARres['LR']['n_regressands']
    k = VARres['LR']['n_regressors']
    m = 1  # from mapping of null hypothesis, here we are testing a pair
    if print_res: print("n, k: ", n, k)

    # Restricted model 1, drop first column
    returns1 = returns.drop(columns=returns.columns[0], axis=1, inplace=False)
    VARres1 = MyVAR.MyVAR(returns1.astype(float), lag)
    if print_res: print(VARres1['LR']['df'].head(3 * lag))
    SSE_R1 = np.diagonal(VARres1['LR']['SS_res'])
    if print_res: print("SSE_R1 ", SSE_R1)

    # Restricted model 2, drop last column
    returns2 = returns.drop(columns=returns.columns[-1], axis=1, inplace=False)
    VARres2 = MyVAR.MyVAR(returns2.astype(float), lag)
    if print_res: print(VARres2['LR']['df'].head(3 * lag))
    SSE_R2 = np.diagonal(VARres2['LR']['SS_res'])
    if print_res: print("SSE_R2 ", SSE_R2)

    # Find critical value of F distribution
    import scipy.stats
    C_alpha = scipy.stats.f.ppf(q = 1-alpha, dfn = m, dfd = (n - k))
    if print_res: print("C_alpha ", C_alpha)

    # Null H_0 is that the betas of the restricted model equal zero (i.e. there is no Granger-causation)

    # A: Test first col does not Granger cause second col:
    # Need to compare the unrestricted model with the
    # model where the first col has been removed, which is VARres1
    LR_A = n * np.log(SSE_R1 / SSE_U[1])
    W_A = (n - k) / m * np.exp(LR_A / n - 1)
    if print_res: print("W_A: ", W_A)
    Reject_Null_A = (W_A > C_alpha)
    if print_res: print("Reject_Null_A? (if True then Granger-causation is present): ", Reject_Null_A)

    # B: Test second col does not Granger cause first col:
    # Need to compare the unrestricted model with the
    # model where the second col has been removed, which is VARres2
    LR_B = n * np.log(SSE_R2 / SSE_U[0])
    W_B = (n - k) / m * np.exp(LR_B / n - 1)
    if print_res: print("W_B: ", W_B)
    Reject_Null_B = (W_B > C_alpha)
    if print_res: print("Reject_Null_B? (if True then Granger-causation is present): ", Reject_Null_B)

    variables = returns.columns
    data = [[1, Reject_Null_B], [Reject_Null_A, 1]]
    df = pd.DataFrame(data=data, columns=variables, index=variables)
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def find_VAR_best_lag(returns, max_lag):
    import MyVAR
    import matplotlib.pyplot as plt
    AIC = pd.DataFrame([MyVAR.MyVAR(returns.astype(float), p+1)['LR']['AIC'] for p in range(max_lag)], index=[p+1 for p in range(max_lag)])
    BIC = pd.DataFrame([MyVAR.MyVAR(returns.astype(float), p+1)['LR']['BIC'] for p in range(max_lag)], index=[p+1 for p in range(max_lag)])
    lags_metrics_df = pd.DataFrame({'AIC': AIC[AIC.columns[0]],
                                'BIC': BIC[BIC.columns[0]]}, index = np.arange(1, max_lag))
    fig, ax = plt.subplots(1, 2, figsize=(15, 3), sharex=True)
    lags_metrics_df.plot(subplots=True, ax=ax, marker='o')
    plt.tight_layout()