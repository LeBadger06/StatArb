import pandas as pd

datapath = "C:\\Users\\Julien.Granger\\PycharmProjects\\JupLab\\data\\Code_Data\\"

def load_ef():

    # NATURAL GAS
    dfng = pd.read_csv(datapath+"NatGas\\Natural_Gas.txt", delimiter ="\t", parse_dates=['Date'], dayfirst=True)
    dfng.dropna(how='all', axis=1, inplace=True)
    dfng['Date'] = pd.to_datetime(dfng['Date'], infer_datetime_format=True)
    dfng.set_index('Date', inplace = True)
    dfng.index = pd.to_datetime(dfng.index)

    dfng1 = pd.read_csv(datapath + "NatGas\\NG_eik.csv", parse_dates=['Date'], dayfirst=False)
    dfng1.dropna(how='all', axis=1, inplace=True)
    dfng1['Date'] = pd.to_datetime(dfng1['Date'], infer_datetime_format=True)
    dfng1.set_index('Date', inplace=True)
    dfng1.index = pd.to_datetime(dfng1.index)

    dfng = pd.concat([dfng, dfng1], axis = 0)
    dfng = dfng[~dfng.index.duplicated(keep='first')]


    # CRUDE
    dfco = pd.read_csv(datapath+"Crude Oil\\Crude_Oil.txt", delimiter ="\t", parse_dates=['Date'], dayfirst=True)
    dfco.dropna(how='all', axis=1, inplace=True)
    dfco['Date'] = pd.to_datetime(dfco['Date'], infer_datetime_format=True)
    dfco.set_index('Date', inplace = True)
    dfco.index = pd.to_datetime(dfco.index)

    dfco1 = pd.read_csv(datapath + "Crude Oil\\CL_eik.csv", parse_dates=['Date'], dayfirst=False)
    dfco1.dropna(how='all', axis=1, inplace=True)
    dfco1['Date'] = pd.to_datetime(dfco1['Date'], infer_datetime_format=True)
    dfco1.set_index('Date', inplace=True)
    dfco1.index = pd.to_datetime(dfco1.index)
    # Correct negative number in CL series
    dfco1.loc['2020-04-20'] = 0.5 * (dfco1.loc['2020-04-17'] + dfco1.loc['2020-04-21'])

    dfco2 = pd.read_csv(datapath + "Crude Oil\\LCO_eik.csv", parse_dates=['Date'], dayfirst=False)
    dfco2.dropna(how='all', axis=1, inplace=True)
    dfco2['Date'] = pd.to_datetime(dfco2['Date'], infer_datetime_format=True)
    dfco2.set_index('Date', inplace=True)
    dfco2.index = pd.to_datetime(dfco2.index)

    dfco12 = pd.concat([dfco1, dfco2], axis = 1)

    dfco = pd.concat([dfco, dfco12], axis=0)
    dfco.fillna(method = 'backfill', inplace = True)
    dfco = dfco[~dfco.index.duplicated(keep='first')]
    dfco.index = pd.to_datetime(dfco.index)

    # MOGAS
    dfmogas = pd.read_csv(datapath+"Gasoline\\Gasoline.txt", delimiter ="\t", parse_dates=['Date'], dayfirst=True)
    dfmogas.dropna(how='all', axis=1, inplace=True)
    dfmogas['Date'] = pd.to_datetime(dfmogas['Date'], infer_datetime_format=True)
    dfmogas.set_index('Date', inplace = True)
    dfmogas.index = pd.to_datetime(dfmogas.index)

    dfmg1 = pd.read_csv(datapath + "Gasoline\\RB_eik.csv", parse_dates=['Date'], dayfirst=False)
    dfmg1.dropna(how='all', axis=1, inplace=True)
    dfmg1['Date'] = pd.to_datetime(dfmg1['Date'], infer_datetime_format=True)
    dfmg1.set_index('Date', inplace=True)
    dfmg1.index = pd.to_datetime(dfmg1.index)

    dfmogas = pd.concat([dfmogas, dfmg1], axis=0)
    dfmogas = dfmogas[~dfmogas.index.duplicated(keep='first')]


    # HEATING OIL
    dfho = pd.read_csv(datapath+"HeatOil\\Heating_Oil.txt", delimiter ="\t", parse_dates=['Date'], dayfirst=True)
    dfho.dropna(how='all', axis=1, inplace=True)
    dfho['Date'] = pd.to_datetime(dfho['Date'], infer_datetime_format=True)
    dfho.set_index('Date', inplace = True)
    dfho.index = pd.to_datetime(dfho.index)

    dfho1 = pd.read_csv(datapath + "HeatOil\\Heating_Oil_eik.csv", parse_dates=['Date'], dayfirst=False)
    dfho1.dropna(how='all', axis=1, inplace=True)
    dfho1['Date'] = pd.to_datetime(dfho1['Date'], infer_datetime_format=True)
    dfho1.set_index('Date', inplace=True)
    dfho1.index = pd.to_datetime(dfho1.index)

    dfho = pd.concat([dfho, dfho1], axis=0)
    dfho = dfho[~dfho.index.duplicated(keep='first')]


    # GASOIL
    dfgo = pd.read_csv(datapath+"Gasoil\\Gasoil.csv", parse_dates=['Date'], dayfirst=False)
    dfgo.dropna(how='all', axis=1, inplace=True)
    dfgo['Date'] = pd.to_datetime(dfgo['Date'], infer_datetime_format=True)
    dfgo.set_index('Date', inplace = True)
    dfgo.index = pd.to_datetime(dfgo.index)

    dfgo1 = pd.read_csv(datapath + "Gasoil\\Gasoil_eik.csv", parse_dates=['Date'], dayfirst=False)
    dfgo1.dropna(how='all', axis=1, inplace=True)
    dfgo1['Date'] = pd.to_datetime(dfgo1['Date'], infer_datetime_format=True)
    dfgo1.set_index('Date', inplace=True)
    dfgo1.index = pd.to_datetime(dfgo1.index)

    dfgo = pd.concat([dfgo, dfgo1], axis=0)
    dfgo = dfgo[~dfgo.index.duplicated(keep='first')]
    
    
    

    df = dict()
    df['NG'] = dfng
    df['RB'] = dfmogas
    df['HO'] = dfho
    df['CR'] = dfco
    df['GO'] = dfgo

    # convert to dollar per bbl
    df['RB'] = 42*df['RB']
    df['HO'] = 42*df['HO']

    # for natural gas to bbl need to divide by 172.4 b/c 10,000 mmBtu = 1,724 Bbl and the contract size is 1,000 bbl for others
    #df['NG'] = df['NG']/172.4
    # wondering about numerical instability though so not doing it for now

    return df

def make_timespread_pairs():
    import LoadEnergyFutures
    import MyPair as MyPair
    df = LoadEnergyFutures.load_ef()
    PairsList = [

                    MyPair.MyPair(df['NG']['NNGC1'], df['NG']['NNGC3']),
                    MyPair.MyPair(df['NG']['NNGC3'], df['NG']['NNGC6']),
                    MyPair.MyPair(df['NG']['NNGC1'], df['NG']['NNGC6']),
                    MyPair.MyPair(df['NG']['NNGC1'], df['NG']['NNGC12']),

                    MyPair.MyPair(df['RB']['RBC1'], df['RB']['RBC3']),
                    MyPair.MyPair(df['RB']['RBC3'], df['RB']['RBC6']),
                    MyPair.MyPair(df['RB']['RBC1'], df['RB']['RBC6']),
                    MyPair.MyPair(df['RB']['RBC1'], df['RB']['RBC12']),

                    MyPair.MyPair(df['HO']['HOTC1'], df['HO']['HOTC3']),
                    MyPair.MyPair(df['HO']['HOTC3'], df['HO']['HOTC6']),
                    MyPair.MyPair(df['HO']['HOTC1'], df['HO']['HOTC6']),
                    MyPair.MyPair(df['HO']['HOTC1'], df['HO']['HOTC12']),

                    MyPair.MyPair(df['CR']['RCLC1'], df['CR']['RCLC3']),
                    MyPair.MyPair(df['CR']['RCLC3'], df['CR']['RCLC6']),
                    MyPair.MyPair(df['CR']['RCLC1'], df['CR']['RCLC6']),
                    MyPair.MyPair(df['CR']['RCLC1'], df['CR']['RCLC12']),

                    MyPair.MyPair(df['CR']['LLCC1'], df['CR']['LLCC3']),
                    MyPair.MyPair(df['CR']['LLCC3'], df['CR']['LLCC6']),
                    MyPair.MyPair(df['CR']['LLCC1'], df['CR']['LLCC6']),

                    MyPair.MyPair(df['GO']['GOC1'], df['GO']['GOC3']),
                    MyPair.MyPair(df['GO']['GOC3'], df['GO']['GOC6']),
                    MyPair.MyPair(df['GO']['GOC1'], df['GO']['GOC6']),
                    MyPair.MyPair(df['GO']['GOC1'], df['GO']['GOC12']),
    ]
    
    PairNames = ['NG13',
                 'NG36',
                 'NG16',
                 'NG112',
                 'RB13',
                 'RB36',
                 'RB16',
                 'RB112',
                 'HO13',
                 'HO36',
                 'HO16',
                 'HO112',
                 'CL13',
                 'CL36',
                 'CL16',
                 'CL112',
                 'LCO13',
                 'LCO36',
                 'LCO16',
                 'GO13',
                 'GO36',
                 'GO16',
                 'GO112'
                 ]
    Timespread_PairsList = [PairNames, PairsList]
    return Timespread_PairsList

def make_summary_steps_123(Timespread_PairsList):
    half_lives = pd.DataFrame([Timespread_PairsList[1][i].halflife_days() for i in range(len(Timespread_PairsList[0]))],
                              index=[i for i in range(len(Timespread_PairsList[0]))])

    step1adf = pd.DataFrame([Timespread_PairsList[1][i].Step1_ADF_tstat() for i in range(len(Timespread_PairsList[0]))],
                            index=[i for i in range(len(Timespread_PairsList[0]))])

    step2ec = pd.DataFrame([Timespread_PairsList[1][i].Step2_ec_tstat() for i in range(len(Timespread_PairsList[0]))],
                           index=[i for i in range(len(Timespread_PairsList[0]))])

    coivec0 = pd.DataFrame([Timespread_PairsList[1][i].CoiVec()['1'] for i in range(len(Timespread_PairsList[0]))],
                           index=[i for i in range(len(Timespread_PairsList[0]))])

    coivec1 = pd.DataFrame([Timespread_PairsList[1][i].CoiVec()['b_2'][0] for i in range(len(Timespread_PairsList[0]))],
                           index=[i for i in range(len(Timespread_PairsList[0]))])

    coivec2 = pd.DataFrame([Timespread_PairsList[1][i].CoiVec()['b_0'][0] for i in range(len(Timespread_PairsList[0]))],
                           index=[i for i in range(len(Timespread_PairsList[0]))])

    theta = pd.DataFrame([Timespread_PairsList[1][i].ou_theta() for i in range(len(Timespread_PairsList[0]))],
                           index=[i for i in range(len(Timespread_PairsList[0]))])

    mu = pd.DataFrame([Timespread_PairsList[1][i].ou_mu() for i in range(len(Timespread_PairsList[0]))],
                         index=[i for i in range(len(Timespread_PairsList[0]))])

    sigma = pd.DataFrame([Timespread_PairsList[1][i].ou_sigma_eq() for i in range(len(Timespread_PairsList[0]))],
                         index=[i for i in range(len(Timespread_PairsList[0]))])

    GC12 = pd.DataFrame([Timespread_PairsList[1][i].gc().iloc[0][0] for i in range(len(Timespread_PairsList[0]))],
                         index=[i for i in range(len(Timespread_PairsList[0]))])

    GC21 = pd.DataFrame([Timespread_PairsList[1][i].gc().iloc[1][1] for i in range(len(Timespread_PairsList[0]))],
                         index=[i for i in range(len(Timespread_PairsList[0]))])

    b2_Joh = pd.DataFrame([Timespread_PairsList[1][i].get_coiVec_Johansen()['b_2'][0] for i in range(len(Timespread_PairsList[0]))],
                           index=[i for i in range(len(Timespread_PairsList[0]))])

    b0_Joh = pd.DataFrame(
        [Timespread_PairsList[1][i].get_coiVec_Johansen()['b_0'][0] for i in range(len(Timespread_PairsList[0]))],
        index=[i for i in range(len(Timespread_PairsList[0]))])

    lag_bic = pd.DataFrame(
        [Timespread_PairsList[1][i].get_lag_Johansen() for i in range(len(Timespread_PairsList[0]))],
        index=[i for i in range(len(Timespread_PairsList[0]))])


    summary = pd.concat([half_lives, step1adf, step2ec, coivec0, coivec1, coivec2, theta, mu, sigma, GC12, GC21, b2_Joh, b0_Joh, lag_bic], axis=1, ignore_index=True)
    summary.columns = ['half_life', 'step1_ADF_tstat', 'step2_ec_tstat', '1', 'b_2', 'b_0', 'OU_theta', 'OU_mu', 'OU_sigma_eq', 'Granger-cause 1->2', 'Granger-cause 2->1', 'b2_joh', 'b0_joh', 'lag_bic_joh']
    summary['spr_name'] = Timespread_PairsList[0]
    return summary

def make_cracks_pairs():
    import LoadEnergyFutures
    import MyPair as MyPair
    df = LoadEnergyFutures.load_ef()
    dfcrho = pd.concat([df['CR']['RCLC1'],df['CR']['RCLC3'],df['CR']['RCLC6'], df['HO']['HOTC1'], df['HO']['HOTC3'], df['HO']['HOTC6']], axis=1)
    dfcrho.dropna(inplace=True)
    dfcrrb = pd.concat([df['CR']['RCLC1'],df['CR']['RCLC3'],df['CR']['RCLC6'], df['RB']['RBC1'], df['RB']['RBC3'], df['RB']['RBC6']], axis=1)
    dfcrrb.dropna(inplace=True)
    dfcrgo = pd.concat([df['CR']['LLCC1'], df['CR']['LLCC3'], df['CR']['LLCC6'], df['GO']['GOC1'], df['GO']['GOC3'], df['GO']['GOC6']],
        axis=1)
    dfcrgo.dropna(inplace=True)
    PairsList = [

                    MyPair.MyPair(dfcrho['RCLC1'], dfcrho['HOTC1']),
                    MyPair.MyPair(dfcrho['RCLC3'], dfcrho['HOTC3']),
                    MyPair.MyPair(dfcrho['RCLC6'], dfcrho['HOTC6']),

                    MyPair.MyPair(dfcrrb['RCLC1'], dfcrrb['RBC1']),
                    MyPair.MyPair(dfcrrb['RCLC3'], dfcrrb['RBC3']),
                    MyPair.MyPair(dfcrrb['RCLC6'], dfcrrb['RBC6']),

                    MyPair.MyPair(dfcrgo['LLCC1'], dfcrgo['GOC1']),
                    MyPair.MyPair(dfcrgo['LLCC3'], dfcrgo['GOC3']),
                    MyPair.MyPair(dfcrgo['LLCC6'], dfcrgo['GOC6'])
    ]

    PairNames = ['CLHOC1',
                 'CLHOC3',
                 'CLHOC6',
                 'CLRBC1',
                 'CLRBC3',
                 'CLRBC6',
                 'LCOGOC1',
                 'LCOGOC3',
                 'LCOGOC6']
    Cracks_PairsList = [PairNames, PairsList]
    return Cracks_PairsList

def make_locational_pairs():
    import LoadEnergyFutures
    import MyPair as MyPair
    df = LoadEnergyFutures.load_ef()
    dfhogo = pd.concat([df['HO']['HOTC2'],df['HO']['HOTC4'],df['HO']['HOTC7'], df['GO']['GOC1'], df['GO']['GOC3'], df['GO']['GOC6']], axis=1)
    dfhogo.dropna(inplace=True)
    PairsList = [

                    MyPair.MyPair(df['CR']['RCLC1'], df['CR']['LLCC2']),
                    MyPair.MyPair(df['CR']['RCLC3'], df['CR']['LLCC4']),

                    MyPair.MyPair(dfhogo['HOTC2'], dfhogo['GOC1']),
                    MyPair.MyPair(dfhogo['HOTC4'], dfhogo['GOC3']),
                    MyPair.MyPair(dfhogo['HOTC7'], dfhogo['GOC6'])
    ]

    PairNames = ['CL1LCO2',
                 'CL3LCO4',
                 'HO2GO1',
                 'HO4GO3',
                 'HO7GO6']
    Location_PairsList = [PairNames, PairsList]
    return Location_PairsList



