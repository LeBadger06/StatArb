import pandas as pd
import numpy as np

def prepare_df(df):
    dg = df.copy()
    dg.columns = ['spread']
    dg['spread_shifted'] = dg['spread'].shift(+1)
    dg['spread_return'] = dg['spread'].diff(+1)
    dg.dropna(inplace=True)
    dg['spr_vol'] = dg['spread'].rolling(window=20).std() # used for stop loss, optional
    #dg.dropna(inplace = True)
    dg.fillna(0, inplace=True)
    return dg

def make_entries_exits(df, level):
    # assume df has column named spread and spread_shifted to avoid data snooping
    # df['spread_return'] = df['spread'].diff(+1)
    # df.dropna(inplace = True)
    dg = df.copy()

    levelup = level
    leveldn = -levelup

    mask_crossdn = (dg['spread_shifted'] >= levelup) & (dg['spread'] < levelup)
    mask_crossup = (dg['spread_shifted'] <= leveldn) & (dg['spread'] > leveldn)
    mask_crosszero_1 = (dg['spread_shifted'] >= dg['spread'].mean()) & (dg['spread'] < dg['spread'].mean())
    mask_crosszero_2 = (dg['spread_shifted'] <= dg['spread'].mean()) & (dg['spread'] > dg['spread'].mean())
    dg['long_enter'] = np.where(mask_crossup, 1, 0)
    dg['short_enter'] = np.where(mask_crossdn, -1, 0)
    dg['cross_zero'] = np.where(mask_crosszero_1 | mask_crosszero_2, 1, 0)

    return dg

def generate_pnl(df, stop_factor = 1, use_stop_loss = False, print_trades = False):
    pd.options.mode.chained_assignment = None

    # assumes df has entries and exits
    # already defined through the function make_entries_exits() above
    current_position_open = False
    entry_level = 0

    df['long_short'] = 0
    df['stop'] = 0
    df['profits'] = 0

    stop_losses = []
    open_positions = []
    closed_positions = []
    profits = []
    long_short = np.zeros(shape=(len(df), 1))
    counter = 0

    pnl_results = dict()

    for i in range(0, len(df)):
        if current_position_open:
            df['stop'][i] = df['stop'][i - 1]
            df['long_short'][i] = df['long_short'][i - 1]

            if (use_stop_loss and
                    (
                            (df['long_short'][i] == 1 and df['spread'][i] < df['stop'][i])
                            or (df['long_short'][i] == -1 and df['spread'][i] > df['stop'][i])
                    )
            ):
                # Stop loss
                current_position_open = False
                profits.append(-np.abs(df['spread'][i] - entry_level))
                closed_positions.append(i)
                long_short[i] = 0

                if print_trades:
                    print('Stop loss at: ', df.index[i])
                    print('Profit is:', np.abs(-np.abs(df['spread'][i] - entry_level)))

            elif df['cross_zero'][i] == 1:
                # Close with profit
                current_position_open = False
                profits.append(np.abs(df['spread'][i] - entry_level))
                closed_positions.append(i)
                long_short[i] = 0

                if print_trades:
                    print('Close w profit at: ', df.index[i])
                    print('Profit is:', np.abs(df['cross_zero'][i] - entry_level))
            else:
                # Keep position open
                df['long_short'][i] = df['long_short'][i - 1]  # 1 if long, -1 if short
                # df['stop'][i] = df['stop'][i-1]
                profits.append(0)
        else:
            if (df['long_enter'][i] == 1) or (df['short_enter'][i] == -1):
                # Enter position
                counter += 1

                if print_trades:
                    print(counter)
                    print('Enter position at: ', df.index[i])

                current_position_open = True
                entry_level = df['spread'][i]

                if (df['long_enter'][i] == 1):
                    df['long_short'][i] = 1
                    df['stop'].iloc[i] = entry_level - stop_factor * (df['spr_vol'][i])

                    if print_trades:
                        print("Rolling vol is: ", df['spr_vol'][i])
                        print("Position is: long, with stop set at: ", df['stop'].iloc[i])

                elif (df['short_enter'][i] == -1):
                    df['long_short'][i] = -1
                    df['stop'].iloc[i] = entry_level + stop_factor * (df['spr_vol'][i])

                    if print_trades:
                        print("Rolling vol is: ", df['spr_vol'][i])
                        print("Position is: short, with stop set at: ", df['stop'].iloc[i])

                open_positions.append(i)
            profits.append(0)  # necessary to make the profits series the same length at the original df

    df.profits = pd.Series(index=df.index, data=profits, dtype = float)

    df['long_short'].fillna(0, inplace=True)
    df['spread_return'].fillna(0, inplace=True)

    # Note that the original price levels were transformed into logs so the below should work
    # Also the price level normalization should not impact the below
    df['backtest'] = df['spread_return'] * df['long_short']

    pnl_results['df'] = df
    pnl_results['open_positions'] = open_positions
    pnl_results['closed_positions'] = closed_positions

    return pnl_results

def run_trading_results(df, level):
    import Backtest
    dg = Backtest.make_entries_exits(df, level)
    pnl_results = Backtest.generate_pnl(dg, use_stop_loss=False)

    import empyrical
    sharpe = empyrical.sharpe_ratio(pnl_results['df']['backtest'])
    maxdd = empyrical.max_drawdown(pnl_results['df']['backtest'])
    tradingResults = dict()
    tradingResults['sharpe'] = sharpe
    tradingResults['max_dd'] = maxdd
    return tradingResults

def optimize_Z(my_pair):
    import Backtest
    dg = Backtest.prepare_df(my_pair.S1s2())
    search_Za = np.zeros((10, 2))
    index = np.zeros(10)
    for i in range(1, 11, 1):
        Z = 0.1 * i
        index[i - 1] = Z
        myTR = Backtest.run_trading_results(dg, Z * my_pair.OU_sigma_eq)
        search_Za[i - 1][0] = myTR['sharpe']
        search_Za[i - 1][1] = myTR['max_dd']

    search_Z = pd.DataFrame(index=index, data=search_Za, columns=['sharpe', 'max_dd'])
    return search_Z


def make_insample_index(pair, yr_factor=4, increment_factor=0.5, print_debug = False):
    from datetime import timedelta
    # input: cointegrated pair
    # yr_factor: how many years for insample window
    # increment_factor: how to stack insample windows, typically 6 months

    numyrs = int(np.round(len(pair.S1s2()) / (252), 0)) - 1

    total_loop = 0
    total_loop = int(1 + (numyrs - yr_factor) / (increment_factor))
    if print_debug: print((numyrs - yr_factor) / (increment_factor))
    if print_debug: print(total_loop)

    if print_debug: print(numyrs)
    insample_index = pd.DataFrame(columns=['Start', 'End'], index=range(0, total_loop + 1))
    for i in range(0, total_loop + 1):
        # print("i=.....", i)
        if i == 0:
            insample_index['Start'].iloc[i] = pair.S1s2().index[0]
            insample_index['End'].iloc[i] = pair.S1s2().index[0] + timedelta(days=yr_factor * 365)
        elif i < total_loop:
            insample_index['Start'].iloc[i] = pd.to_datetime(insample_index['Start'][i - 1]) + timedelta(
                days=increment_factor * 365)
            insample_index['End'].iloc[i] = pd.to_datetime(insample_index['Start'][i]) + timedelta(days=yr_factor * 365)
        else:
            insample_index['Start'].iloc[i] = pd.to_datetime(insample_index['Start'][i - 1]) + timedelta(
                days=increment_factor * 365)
            insample_index['End'].iloc[i] = pair.S1s2().index[-1]

    return insample_index

def make_insample_pairs(insample_index, df, col1, col2):
    import MyPair
    # Given an original df and an index of insample time windows
    # we look whether the series in the window are cointegrated
    pairs_list_in_sample = []
    for i in range(0, len(insample_index)):
        start = insample_index['Start'][i]
        stop = insample_index['End'][i]
        newpair = MyPair.MyPair(df[col1][start:stop], df[col2][start:stop])
        pairs_list_in_sample.append(newpair)
    return pairs_list_in_sample


def make_outsample_index(insample_index, increment_factor=0.5):
    from datetime import timedelta

    outsample_index = pd.DataFrame(columns=['Start', 'End'], index=range(0, len(insample_index)))

    for i in range(0, len(insample_index)):
        outsample_index['Start'].iloc[i] = insample_index['End'].iloc[i]
        outsample_index['End'].iloc[i] = outsample_index['Start'].iloc[i] + timedelta(
            days=increment_factor * 365)

    return outsample_index


def make_outofsample_params(out_of_sample_index, b2_is, b0_is, OU_sigma_eq_is):
    # This function makes out of sample b_2 and b_0 and OU_sigma
    # which will be used for backtesting later on

    from datetime import timedelta

    df = pd.DataFrame(columns=['Start', 'End', 'b2_is', 'b0_is', 'OU_sigma_eq_is', 'New_index'])

    for i in range(0, len(out_of_sample_index)):
        newrows = pd.DataFrame(index=range(0, 2), columns=['Start', 'End', 'b2_is', 'b0_is', 'OU_sigma_eq_is', 'New_index'])

        newrows['Start'].iloc[0] = pd.to_datetime(out_of_sample_index['Start'].iloc[i])
        newrows['End'].iloc[0] = pd.to_datetime(out_of_sample_index['End'].iloc[i])
        newrows['b2_is'].iloc[0] = b2_is.iloc[i]['b_2']
        newrows['b0_is'].iloc[0] = b0_is.iloc[i]['b_0']
        newrows['OU_sigma_eq_is'].iloc[0] = OU_sigma_eq_is.iloc[i]['OU_sigma_eq']
        newrows['New_index'].iloc[0] = newrows['Start'].iloc[0]  # This will be the index

        newrows['Start'].iloc[1] = pd.to_datetime(out_of_sample_index['Start'].iloc[i])
        newrows['End'].iloc[1] = pd.to_datetime(out_of_sample_index['End'].iloc[i])
        newrows['b2_is'].iloc[1] = b2_is.iloc[i]['b_2']
        newrows['b0_is'].iloc[1] = b0_is.iloc[i]['b_0']
        newrows['OU_sigma_eq_is'].iloc[1] = OU_sigma_eq_is.iloc[i]['OU_sigma_eq']
        newrows['New_index'].iloc[1] = newrows['End'].iloc[0] + timedelta(
            days=-1)  # This will be the index

        df = pd.concat([df, newrows], axis=0, join='inner')

    df['New_index'] = pd.to_datetime(df['New_index'])
    df.rename(columns={'New_index': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True, drop=True)
    df.drop(['Start', 'End'], axis=1, inplace=True)
    dfd = df.resample('d').bfill()
    return dfd

def make_outofsample_spread(dff, s1, s2):
    # This function makes out of sample spreads
    # given out of sample b_2 and b_0 and price series s1 and s2

    # Need to normalize first
    s1 = s1 / s1.iloc[0]
    s2 = s2 / s2.iloc[0]

    dg = pd.concat([dff, s1, s2], axis = 1, join = 'outer')
    dg.dropna(inplace = True)

    # spr = 1 * s1 + b_2 * s2 + b_0
    dg['spread'] = dg[dg.columns[-2]] + dg['b2_is']*dg[dg.columns[-1]] + dg['b0_is']
    return dg


def run_trading_results_outofsample(os_index, os_sp, Z=0.3, use_stop_loss=False, print_trades=False):
    # arguments:
    # os_index: out os sample index
    # os_sp: out of sample spread, which also contains the in sample OU parameters such as

    # Disable warning
    pd.options.mode.chained_assignment = None

    import Backtest

    df_all = pd.DataFrame()

    for i in range(0, len(os_index)):
        # slice the spread
        dgos_i = os_sp[os_index["Start"][i]:os_index["End"][i]]

        # prepare dgos_i
        dgos_i['spread_shifted'] = dgos_i['spread'].shift(+1)
        dgos_i['spread_return'] = dgos_i['spread'].diff(+1)
        dgos_i['spr_vol'] = dgos_i['spread'].rolling(window=20).std()  # used for stop loss, optional
        dgos_i.dropna(inplace=True)

        # make entries and exits
        level = Z * dgos_i['OU_sigma_eq_is']
        dg_entries_exits_i = Backtest.make_entries_exits(dgos_i, level)

        # generate pnl
        pnl_results_i = Backtest.generate_pnl(dg_entries_exits_i, use_stop_loss, print_trades)

        # concatenate df_all to store the results
        dfi = pnl_results_i['df']
        df_all = pd.concat([df_all, dfi], axis=0, join='outer')



    # Drop duplicate indices
    df_all = df_all[~df_all.index.duplicated(keep='first')]

    df_all.fillna(0, inplace=True)
    #df_all.dropna(inplace=True)

    import empyrical
    sharpe = empyrical.sharpe_ratio(df_all['backtest'])
    maxdd = empyrical.max_drawdown(df_all['backtest'])
    tradingResults = dict()
    tradingResults['sharpe'] = sharpe
    tradingResults['max_dd'] = maxdd
    tradingResults['df'] = df_all

    return tradingResults

def plot_results(pair, Z = 0.3, use_stop_loss = False):
    import Backtest
    dg = Backtest.prepare_df(pair.S1s2())
    levelup = Z * pair.OU_sigma_eq
    dg = Backtest.make_entries_exits(dg, levelup)
    pnl_results = Backtest.generate_pnl(dg, use_stop_loss)
    df1 = pnl_results['df']
    open_positions = pnl_results['open_positions']
    closed_positions = pnl_results['closed_positions']

    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (20, 10)
    leveldn = -Z * pair.OU_sigma_eq
    mu = pair.OU_mu
    plt.plot_date([df1['spread'].index[i] for i in open_positions], [df1['spread'][i] for i in open_positions],
                  label='Open position', marker='^', markeredgecolor='g', markerfacecolor='b', markersize=6)
    plt.plot_date([df1['spread'].index[i] for i in closed_positions], [df1['spread'][i] for i in closed_positions],
                  label='Closed position', marker='P', markeredgecolor='r', markerfacecolor='r', markersize=6)
    dg.spread_shifted.plot()
    plt.axhline(y=levelup, color='r', linestyle='--')
    plt.axhline(y=leveldn, color='r', linestyle='--')
    plt.axhline(y=mu, color='b', linestyle='--')

    import pyfolio as pf
    pf.create_returns_tear_sheet(df1['backtest'].astype(float))

def rolling_oos_backtesting_plots(dfsel, col1, col2, estimation_window = 3, out_of_sample_window = 0.33, Z = 0.3, use_stop_loss = False):
    import MyPair
    import Backtest
    import pyfolio as pf
    import pandas as pd
    import matplotlib.pyplot as plt

    s1 = dfsel[col1]
    s2 = dfsel[col2]

    this_pair = MyPair.MyPair(s1, s2)

    s1_norm = this_pair.S1()
    s2_norm = this_pair.S2()

    in_sample_index = Backtest.make_insample_index(this_pair, estimation_window, out_of_sample_window)

    out_sample_index = Backtest.make_outsample_index(in_sample_index)

    pairs_list_in_sample = Backtest.make_insample_pairs(in_sample_index, dfsel, col1, col2)

    tstat_adf_is = pd.DataFrame([pairs.step1_ADF_tstat for pairs in pairs_list_in_sample], columns={'tstat_ADF'},
                            index=in_sample_index["End"])
    tstat_adf_is_rsampled = tstat_adf_is.resample('d').bfill()

    tstat_ec_is = pd.DataFrame([pairs.Step2_ec_tstat()[0] for pairs in pairs_list_in_sample], columns={'tstat_ec'},
                               index=in_sample_index["End"])
    tstat_ec_is_rsampled = tstat_ec_is.resample('d').bfill()

    b2_is = pd.DataFrame([pairs.CoiVec()['b_2'][0] for pairs in pairs_list_in_sample], columns={'b_2'},
                         index=in_sample_index["End"])
    b2_is_rsampled = b2_is.resample('d').bfill()

    b0_is = pd.DataFrame([pairs.CoiVec()['b_0'][0] for pairs in pairs_list_in_sample], columns={'b_0'},
                         index=in_sample_index["End"])

    OU_sigma_eq_is = pd.DataFrame([pairs.ou_sigma_eq() for pairs in pairs_list_in_sample], columns={'OU_sigma_eq'},
                                  index=in_sample_index["End"])

    dff = Backtest.make_outofsample_params(out_sample_index, b2_is, b0_is, OU_sigma_eq_is)

    spread_os = Backtest.make_outofsample_spread(dff, s1, s2)

    # Trading Results
    TR = Backtest.run_trading_results_outofsample(out_sample_index, spread_os, Z, use_stop_loss,
                                                  print_trades=True)

    pf.create_returns_tear_sheet(TR['df']['backtest'].astype(float))


    output_to_plot = pd.concat([s1_norm, s2_norm, tstat_adf_is_rsampled, tstat_ec_is_rsampled, b2_is_rsampled, spread_os.spread], axis = 1)
    #output_to_plot.dropna(inplace = True)
    fig, ax = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    output_to_plot.plot(subplots=True, ax=ax)

    plt.tight_layout()