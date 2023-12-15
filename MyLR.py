import numpy as np
import pandas as pd


def linear_regression(Y, X, print_debug = False):
    ##
    ## EXPECTS dataframes, not arrays, because it needs the names of columns for reporting purposes
    ##
    X_cols = X.columns
    if print_debug: print(X_cols)
    Y_cols = Y.columns
    if print_debug: print(Y_cols)

    Y = Y.to_numpy()
    X = X.to_numpy()

    X_T = X.T
    X_T_X = X_T @ X

    # observations m, number of daily bars
    m = Y.shape[0]
    # regressands n
    n = Y.ndim
    # regressors l
    l = X.ndim
    if print_debug: print('m, n, l :', m, n, l)

    LR_results = dict()
    LR_results['n_regressands'] = m
    LR_results['n_regressors'] = X.shape[1]

    rank = np.linalg.matrix_rank(X_T_X)

    if rank == 1:
        X_T_X_inv = 1 / X_T_X
        B_hat = X_T_X_inv * X_T @ Y
        residuals = Y - B_hat * X
    elif rank > 1:
        try:
            X_T_X_inv = np.linalg.inv(X_T_X)
        except np.linalg.LinAlgError:
            print("Singular matrix XtX.")

        B_hat = (X_T_X_inv) @ (X_T @ Y)
        residuals = Y - (X @ B_hat)

    LR_results['B_hat'] = B_hat
    LR_results['residuals'] = residuals

    residual_covariance_matrix = (1 / m) * (residuals.T @ residuals)
    if print_debug: print('residual_covariance_matrix ', residual_covariance_matrix)

    covariance_matrix_regression_coeff = np.kron(residual_covariance_matrix, X_T_X_inv)
    if print_debug: print('covariance_matrix_regression_coeff ', covariance_matrix_regression_coeff)

    LR_results['cov_matr'] = covariance_matrix_regression_coeff

    # Compute R^2, how much variance is explained by the data
    SS_tot = Y.T @ Y
    SS_res = residuals.T @ residuals
    R_sq_unc = 1 - SS_res /  SS_tot
    LR_results['R_sq_unc'] = R_sq_unc
    LR_results['SS_res'] = SS_res

    factor = 1
    if m - l > 0:
        factor = m / (m - l)

    if rank == 1:
        standard_error_regression_coeff = np.sqrt(factor * covariance_matrix_regression_coeff)
    elif rank > 1:
        standard_error_regression_coeff = np.sqrt(factor * np.diagonal(covariance_matrix_regression_coeff))

    LR_results['stderr_coef'] = standard_error_regression_coeff

    standard_error_regression_coeff_reshape = np.reshape(standard_error_regression_coeff, (Y.shape[1], X.shape[1])).T

    LR_results['stderr_coef_reshape'] = standard_error_regression_coeff_reshape

    if print_debug: print('standard_error_regression_coeff', standard_error_regression_coeff)
    if print_debug: print('standard_error_regression_coeff', standard_error_regression_coeff.shape)

    tstat = B_hat / standard_error_regression_coeff_reshape
    if print_debug: print('tstat ', tstat)

    LR_results['tstat'] = tstat

    my_det = residual_covariance_matrix
    if residual_covariance_matrix.ndim > 1:
        my_det = np.linalg.det(residual_covariance_matrix)

    AIC = np.log(my_det) + 2 * l * n / m
    LR_results['AIC'] = AIC

    BIC = np.log(my_det) + np.log(m) * l * n / m
    LR_results['BIC'] = BIC

    coefdf = pd.DataFrame(data=B_hat, columns=pd.MultiIndex.from_product([['Estimate'],
                                         Y_cols]), index=X_cols)
    stderrdf = pd.DataFrame(data=standard_error_regression_coeff_reshape, columns= pd.MultiIndex.from_product([['SD of Estimate'],
                                         Y_cols]), index=X_cols)
    tstatdf= pd.DataFrame(data=tstat, columns= pd.MultiIndex.from_product([['t-Statistic'],
                                         Y_cols]), index=X_cols)
    resultdf = pd.concat([coefdf, stderrdf, tstatdf], axis = 1)

    LR_results['df'] = resultdf

    return LR_results


def add_constant(X):
    # Insert an intercept column
    X.insert(0, "Intercept", 1)
    return X

def add_trend(X):
    # Insert a trend column
    # Assumes X is a df sorted in chronological order (oldest at top)

    # regressors l
    l = X.shape[0]

    trend = np.array(np.ones(l))
    for i in range(0, l):
        trend[i] = (l-i)*trend[i]

    X.insert(0, "Trend", trend)
    return X