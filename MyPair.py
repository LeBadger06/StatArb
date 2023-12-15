import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar import vecm
import MyVAR

class MyPair:

    def __init__(self, s1, s2):
        # Cointegration is sensitive to initial conditions
        # take the log prices then normalize

        log_s1 = np.log(s1)
        log_s2 = np.log(s2)

        self.s1_norm = log_s1/log_s1.iloc[0]
        self.s2_norm = log_s2/log_s2.iloc[0]

        self.df_levels = pd.concat([self.s1_norm, self.s2_norm], axis=1)

        # Engle-Granger
        self.s1s2 = []
        self.step1_ADF_tstat = 0
        self.step2_ec_tstat = 0
        self.coiVec = []
        self.shortTerm = []
        self.errCorr = []
        self.step1Test = False
        self.isECsig = False

        self.OU_theta = 0
        self.OU_mu = 0
        self.OU_sigma_eq = 0
        self.halflife = 0
        self.GC = 0


        # VECM using Johansen approach
        self.s1s2_vecm = 0
        self.coiVec_vecm = 0
        self.lag_vecm = 0

        # Run functions
        self.Run_Engle_Granger()
        self.Fit_to_OU()
        self.Run_Granger_Causality()

        self.Run_Johansen_Approach()



    def Run_Engle_Granger(self):
        # Step 1: obtain the fitted residual $\hat{e_t}$ and test for stationarity, where:
        # P^A_T = \beta_0+\beta_1 P^B_t+\epsilon_t
        # \hat{e}_t = P^A_t -\hat{\beta}_C P^B_t - const
        # If the fitted residual is not stationary, then no long-run relationship
        # exists and regression is spurious.
        Y = self.s1_norm
        X = self.s2_norm

        # With statsmodel interface
        #X = sm.add_constant(X)
        #model = sm.OLS(Y.astype(float), X.astype(float))
        #results = model.fit()
        #residuals_s1s2 = self.s1_norm - results.params[1] * self.s2_norm - results.params[0]

        # With MyOLS interface
        import MyLR, MyOLS
        X = pd.DataFrame(self.s2_norm)
        Y1 = pd.DataFrame(self.s1_norm)
        X1 = MyLR.add_constant(X)
        mymodel = MyOLS.MyOLS(Y1, X1)
        results = mymodel.fit()
        residuals_s1s2 = self.s1_norm - mymodel.params[1] * self.s2_norm - mymodel.params[0]

        # If True residual is stationary
        adfresults = adfuller(residuals_s1s2, 1, regression = "n")
        self.step1_ADF_tstat = adfresults[0]
        self.step1Test = ( self.step1_ADF_tstat  <= adfresults[4]['5%'] )

        self.s1s2 = pd.DataFrame(index = self.s1_norm.index, data = residuals_s1s2)

        # With statsmodel interface
        #self.coiVec = pd.DataFrame([{'1': 1, 'b_2': - results.params[1], 'b_0': results.params[0]}])

        # With MyOLS interface
        self.coiVec = pd.DataFrame([{'1': 1, 'b_2': - mymodel.params[1], 'b_0': mymodel.params[0]}])

        # Step 2: Plug the stationary fitted residual $\hat{e}_{t-1}$ from previous step,
        # shifted, into error correction linear regression and confirm statistical significance of its coefficient.
        # \Delta P^A_t = \phi \Delta P^B_t -(1-\alpha)\hat{e}_{t-1}

        s1_diff = self.s1_norm.diff()
        s1_diff.dropna(inplace=True)
        s1_diff.rename('s1_diff', inplace=True)
        s2_diff = self.s2_norm.diff()
        s2_diff.dropna(inplace=True)
        s2_diff.rename('s2_diff', inplace=True)
        residuals_s1s2_shifted = residuals_s1s2.shift(+1)
        residuals_s1s2_shifted.dropna(inplace=True)
        residuals_s1s2_shifted.rename('residuals_s1s2_shifted', inplace=True)

        # Error correction
        Y = pd.DataFrame(s1_diff)
        X = pd.concat([s2_diff, residuals_s1s2_shifted], axis=1)

        # With statsmodel interface
        #model = sm.OLS(Y, X)
        #results = model.fit()
        #self.shortTerm = results.params[0]
        #self.errCorr = results.params[1]

        # With MyOLS interface
        mymodel = MyOLS.MyOLS(Y, X)
        results = mymodel.fit()
        self.shortTerm = mymodel.params[0]
        self.errCorr = mymodel.params[1]

        # Statistical significance: must have |tstat| greater than 2
        # (actually it depends on degrees of freedom, etc.)
        #self.isECsig = ( np.abs(results.tvalues['residuals_s1s2_shifted']) >= 2.0 )
        #self.step2_ec_tstat = results.tvalues['residuals_s1s2_shifted']
        self.isECsig = (np.abs(results['df']['t-Statistic'].loc['residuals_s1s2_shifted']) >= 2.0)
        self.step2_ec_tstat = results['df']['t-Statistic'].loc['residuals_s1s2_shifted']

    def Step1_ADF_tstat(self):
        return self.step1_ADF_tstat

    def Step2_ec_tstat(self):
        return self.step2_ec_tstat

    def CoiVec(self):
        return self.coiVec

    def S1s2(self):
        return self.s1s2

    def ShortTerm(self):
        return self.shortTerm

    def ErrCorr(self):
        return self.errCorr

    def get_levels(self):
        return self.df_levels

    def Fit_to_OU(self):
        # The linear cointegrating combination $\beta'_C Y_t = e_t$ produces a stationary and mean-reverting spread.
        # Reversion speed $\theta$ and bounds calculated as $\frac{\sigma_{OU}}{\sqrt{2\theta}}$. $e_t$ follows the SDE:
        # $$
        # de_t = -\theta(e_t-\mu)dt+\sigma_{OU}dW_t
        # $$
        # rewritten as:
        # $$
        # e_{t+1} = C + Be_t+\epsilon_t
        # $$
        # Once regression is estimated, we can solve for:
        # $$
        # \theta = -\frac{\ln(B)}{\tau} = -252\times\ln(B)
        # $$
        # $$
        # \mu = \frac{C}{1-B}
        # $$

        # Ornstein-Uhlenbeck
        df = pd.DataFrame()
        df['spread'] = self.s1s2
        df['spread_shifted'] = df['spread'].shift(+1)
        df.dropna(inplace=True)
        Y = df['spread']
        Y.columns = ['spread']
        X = df['spread_shifted']
        X = sm.add_constant(X)
        X.columns = ['Intercept', 'spread_shifted']

        # With statsmodel
        model = sm.OLS(Y.astype(float), X.astype(float))
        results = model.fit()
        C = results.params[0]
        B = results.params[1]

        # With MyOLS
        import MyOLS
        #mymodel = MyOLS.MyOLS(Y, X)
        #print(Y.head())
        #print(Y.shape)
        #print(X.head())
        #results = mymodel.fit()
        #C = mymodel.params[0]
        #B = mymodel.params[1]

        gamma = 1 / 252 # assume daily bars
        self.OU_theta = -np.log(B) / gamma
        self.OU_mu = C / (1 - B)
        self.halflife = np.log(2) / self.OU_theta
        factor = gamma / (1 - np.exp(-2 * self.OU_theta * gamma))

        # Sum of squares of the residuals
        SSres = np.matmul((Y - np.matmul(X, results.params)).T, (Y - np.matmul(X, results.params)))

        self.OU_sigma_eq = np.sqrt(factor * SSres)

    def ou_mu(self):
        return self.OU_mu

    def ou_theta(self):
        return self.OU_theta

    def ou_sigma_eq(self):
        return self.OU_sigma_eq

    def halflife(self):
        return self.halflife

    def halflife_days(self):
        gamma = 1/252
        return self.halflife/gamma

    def S1(self):
        return self.s1_norm

    def S2(self):
        return self.s2_norm

    def gc(self):
        return self.GC

    def get_spread_Johansen(self):
        return self.spread_vecm

    def get_coiVec_Johansen(self):
        return self.coiVec_vecm

    def get_lag_Johansen(self):
        return self.lag_vecm

    def Run_Granger_Causality(self):
        max_gc_lag = 10
        returns = self.df_levels.diff().dropna()
        self.GC = MyVAR.Granger_likelihood_ratio_test(returns, max_gc_lag, print_res=False)

    def Run_Johansen_Approach(self):
        self.df_levels.index = pd.to_datetime(self.df_levels.index).to_period('B')
        lag_order = vecm.select_order(self.df_levels.astype(float), maxlags=8)
        self.lag_vecm = lag_order.bic

        rank_test_results_np = np.arange(4).reshape(2,2)
        rank_test_results_np[0,0] = vecm.select_coint_rank(self.df_levels.astype(float), 0, lag_order.bic, method="trace",
                                                         signif=0.05).rank
        rank_test_results_np[0,1] = vecm.select_coint_rank(self.df_levels.astype(float), 0, lag_order.bic, method="maxeig",
                                                         signif=0.05).rank
        rank_test_results_np[1,0] = vecm.select_coint_rank(self.df_levels.astype(float), 0, lag_order.bic, method="trace",
                                                         signif=0.01).rank
        rank_test_results_np[1,1] = vecm.select_coint_rank(self.df_levels.astype(float), 0, lag_order.bic,
                                                         method="maxeig",
                                                         signif=0.01).rank
        rank_test_results = pd.DataFrame(data = rank_test_results_np, index = ['0.05','0.01'], columns = ['trace','maxeig'])

        # We must have one cointegration relationship, not zero, not two
        ### Code below is dirty but it works
        isJohansenCointegrationPresent0 = (rank_test_results == 1).any()
        isJohansenCointegrationPresent = (isJohansenCointegrationPresent0 == 1).any()
        ###


        if isJohansenCointegrationPresent: ## that is: rank_test.rank = 1

            # We assume a constant inside the cointegration relationship only
            mymodel = vecm.VECM(self.df_levels.astype(float),
                                deterministic = "ci",
                                k_ar_diff = lag_order.bic,
                                coint_rank = 1)

            vecm_res = mymodel.fit()

            self.spread_vecm = vecm_res.beta[0][0] * self.s1_norm + vecm_res.beta[1][0] * self.s2_norm \
                               + vecm_res.det_coef_coint[0][0]

            self.coiVec_vecm = pd.DataFrame(data = {'1':vecm_res.beta[0][0], 'b_2':vecm_res.beta[1][0], 'b_0':vecm_res.det_coef_coint[0][
                0]}, index = [0])
        else:
            self.coiVec_vecm = pd.DataFrame(data={'1': 0, 'b_2': 0, 'b_0': 0}, index=[0])

