import numpy as np

from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class BestSubsetLinearRegression(LinearRegression):

    def __init__(self, k, **kwargs):
        self.k = k
        super().__init__(**kwargs)

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y)
        self.all_mse_ = []
        best_combination, best_mse = None, np.inf
        best_intercep_, best_coef_ = None, None
        for combination in combinations(range(X.shape[1]), self.k):
            if combination:
                super().fit(X[:, combination], y, sample_weight)
                mse = mean_squared_error(y, self.predict(X[:, combination]))
            else:
                self.intercept_ = y.mean()
                self.coef_ = []
                mse = y.var()
            self.all_mse_.append(mse)
            if mse < best_mse:
                best_combination, best_mse = combination, mse
                best_intercept_, best_coef_ = self.intercept_, self.coef_
        super().fit(X, y, sample_weight)
        self.intercept_ = best_intercept_
        self.coef_[:] = 0
        self.coef_[list(best_combination)] = best_coef_
        return self
