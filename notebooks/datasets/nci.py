import pandas as pd

from pathlib import Path


data_dir = Path(__file__).absolute().parents[2] / "data"
labels = pd.read_table(
    data_dir / "nci.label.txt", header=None,
).iloc[:, 0].str.strip().values
X = pd.read_csv(data_dir / "nci.data.csv", index_col=0)
X.columns = labels

del data_dir, labels
