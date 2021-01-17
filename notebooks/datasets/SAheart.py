import pandas as pd

from pathlib import Path


X_cols = ['sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age']
y_col = 'chd'

data = pd.read_csv(
    Path(__file__).absolute().parents[2] / "data" / "SAheart.data",
    usecols=[y_col] + X_cols,
)
data = data.assign(famhist=(data['famhist'] == "Present").astype(int))

X = data[X_cols]
y = data[y_col]
