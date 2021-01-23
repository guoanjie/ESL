import numpy as np

from itertools import combinations, product
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class BestSubsetRegression(LinearRegression):

    def __init__(self, k=None, **kwargs):
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


class PrincipalComponentsRegression(LinearRegression):

    def __init__(self, n_components=None, **kwargs):
        self.n_components = n_components
        super().__init__(**kwargs)

    def fit(self, X, y, sample_weight=None):
        pca = PCA(n_components=self.n_components)
        X = pca.fit_transform(X)
        super().fit(X, y)
        self.intercept_ -= self.coef_ @ pca.components_ @ pca.mean_
        self.coef_ = np.dot(self.coef_, pca.components_)
        return self


class PartialLeastSquares(PLSRegression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        return {
            **super().get_params(deep=deep), 'n_components': self.n_components,
        }

    def fit(self, X, y):
        super().fit(X, y)
        self.intercept_ = self.y_mean_[0]
        self.coef_ = self.coef_.squeeze()
        return self


class ForwardStagewiseRegression(LinearRegression):

    def __init__(self, eps=1e-3):
        self.eps = eps

    def fit(self, X, y):
        X, y = self._validate_data(X, y.copy())
        self.intercept_ = y.mean()
        y -= self.intercept_
        self.coef_ = np.zeros(X.shape[1])
        self.coef_path_ = self.coef_.copy()

        while True:
            corr = np.dot(y, X)
            j = np.argmax(np.abs(corr))
            delta = self.eps * np.sign(corr[j])
            if self.coef_[j] * delta < 0:
                break
            self.coef_[j] += delta
            y -= delta * X[:, j]
            self.coef_path_ = np.vstack([self.coef_path_, self.coef_])


class NaturalCubicSplineBasis(BaseEstimator, TransformerMixin):

    def __init__(self, dof, tensor_product=False):
        self.dof = dof
        self.tensor_product = tensor_product

    def fit(self, X, y=None):
        q = np.linspace(0, 1, self.dof + 1)
        self.knots = []
        for i in range(X.shape[1]):
            self.knots.append(np.unique(np.quantile(X[:, i], q)))
        return self

    def transform(self, X):
        basis = []
        for i in range(X.shape[1]):
            basis.append(self._basis_1d(X[:, i:i+1], self.knots[i]))
        if self.tensor_product:
            basis.append(self._tensor_product(basis))
        return np.hstack(basis)

    @classmethod
    def _basis_1d(cls, X, knots):
        basis = [X]
        dk_last = cls._dk(X, knots[-2], knots[-1])
        for knot in knots[:-2]:
            basis.append(cls._dk(X, knot, knots[-1]) - dk_last)
        return np.hstack(basis)

    @classmethod
    def _dk(cls, X, knot, knot_last):
        return np.power((X - knot).clip(0), 3) / (knot_last - knot)

    @classmethod
    def _tensor_product(cls, basis_1d):
        return np.hstack([
            np.prod(
                [basis_1d[d][:, i:i+1] for d, i in enumerate(indices)], axis=0,
            )
            for indices in product(*[range(b.shape[1]) for b in basis_1d])
        ])
