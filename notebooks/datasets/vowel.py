import pandas as pd

from pathlib import Path


data_dir = Path(__file__).absolute().parents[2] / "data"
df_train = pd.read_csv(data_dir / "vowel.train", index_col='row.names')
df_test  = pd.read_csv(data_dir / "vowel.test" , index_col='row.names')
X_train = df_train.filter(regex='^x\.')
y_train = df_train['y']
X_test = df_test.filter(regex='^x\.')
y_test = df_test['y']

del data_dir
