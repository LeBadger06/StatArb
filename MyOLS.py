import numpy as np
import MyLR

class MyOLS:

    def __init__(self, Y, X):
        self.Results = dict()
        self.Y = Y
        self.X = X
        self.m = Y.shape[0]
        self.params = 0
        self.tvalues = 0

    def fit(self):
        self.Results = MyLR.linear_regression(self.Y, self.X)
        self.params = self.Results['df']['Estimate'].to_numpy()
        self.tvalues = self.Results['df']['t-Statistic'].to_numpy()
        return self.Results

    def Summary(self):
        print(self.Results['df'].head(self.m))

    def Params(self):
        print(self.Results['df']['Estimate'].head(self.m))

    def Tvalues(self):
        print(self.Results['df']['t-Statistic'].head(self.m))
