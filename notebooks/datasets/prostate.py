import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler


X_cols = [
    'lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45',
]
y_col = 'lpsa'

data = pd.read_table(
    Path(__file__).absolute().parents[2] / "data" / "prostate.data",
    index_col=0,
)

is_train = data['train'] == 'T'
data = data[[y_col] + X_cols]

X = pd.DataFrame(
    data=StandardScaler().fit_transform(data[X_cols]),
    index=data.index, columns=X_cols,
)
y = data[y_col]

X_train = X[ is_train]
y_train = y[ is_train]
X_test  = X[~is_train]
y_test  = y[~is_train]

del X_cols, y_col, is_train
