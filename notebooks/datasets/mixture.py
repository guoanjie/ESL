import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import sys
import tempfile

from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

sys.path.append(str(Path(__file__).absolute().parents[1]))
from colors import BLUE, ORANGE


def read_data(var):
    fname = Path(__file__).absolute().parents[2] / "data" / "ESL.mixture.rda"
    with tempfile.NamedTemporaryFile(suffix='.csv') as fp:
        _ = subprocess.check_output(f"""
            R -e "load('{fname}')
            write.csv(ESL.mixture\${var}, '{fp.name}', row.names=FALSE)"
        """, shell=True)
        return pd.read_csv(fp.name).values.squeeze()


x           = read_data('x'         )
y           = read_data('y'         )
xnew        = read_data('xnew'      )
prob        = read_data('prob'      )
marginal    = read_data('marginal'  )
px1         = read_data('px1'       )
px2         = read_data('px2'       )
means       = read_data('means'     )

del read_data


class OptimalBayes(GaussianMixture):

    def __init__(self):
        super().__init__(
            n_components=means.shape[0],
            covariance_type='spherical',
            means_init=means,
        )

    def fit(self, X, y=None):
        super().fit(means)
        self.covariances_ = [.2] * self.n_components
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type,
        )

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        return proba.reshape(proba.shape[0], 2, -1).sum(axis=-1)


def plot(clf):
    xx, yy = np.meshgrid(
        np.linspace(px1.min(), px1.max(), num=len(px1) * 10 - 9),
        np.linspace(px2.min(), px2.max(), num=len(px2) * 10 - 9),
    )

    cm = ListedColormap([BLUE, ORANGE])

    f, ax = plt.subplots()
    clf.fit(x, y)

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    elif hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = (clf.predict(np.c_[xx.ravel(), yy.ravel()]) > .5).astype(float)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.2)

    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm)

    ax.set_xlim(xx.min() - .1, xx.max() + .1)
    ax.set_ylim(yy.min() - .1, yy.max() + .1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect((xx.max() - xx.min()) / (yy.max() - yy.min()))
