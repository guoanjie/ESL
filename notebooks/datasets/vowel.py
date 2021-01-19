import numpy as np
import pandas as pd

from numpy import linalg as LA
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data_dir = Path(__file__).absolute().parents[2] / "data"
df_train = pd.read_csv(data_dir / "vowel.train", index_col='row.names')
df_test  = pd.read_csv(data_dir / "vowel.test" , index_col='row.names')
X_train = df_train.filter(regex='^x\.')
y_train = df_train['y']
X_test = df_test.filter(regex='^x\.')
y_test = df_test['y']

del data_dir

clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X_train, y_train)
w, v = LA.eig(clf.covariance_)
W_rsqrt = v @ np.diag(np.power(w, -.5)) @ v.T
M_ = clf.means_ @ W_rsqrt
B_ = np.cov(M_.T)
D, V_ = LA.eig(B_)
V = - W_rsqrt @ V_

del w, v, W_rsqrt, M_, B_, D, V_

def plot_lda(*coords, ax=None):
    palette = [
        "#000000", "#0718F5", "#963330", "#9133E7", "#ED9135", "#7DFBFD",
        "#74808E", "#FBEC98", "#000000", "#E73323", "#7DFA4C",
    ]
    ax = (X_train @ V).plot.scatter(
        coords[0] - 1, coords[1] - 1, ax=ax, c='none', edgecolors=palette,
    )
    ax = pd.DataFrame(data=clf.means_ @ V).plot.scatter(
        coords[0] - 1, coords[1] - 1, ax=ax, c='none', edgecolors=palette,
        linewidth=5,
    )
    ax.set_xlabel(f"Coordinate {coords[0]}")
    ax.set_ylabel(f"Coordinate {coords[1]}")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Linear Discriminant Analysis")
