import pandas as pd

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


def plot_lda(*coords, ax=None):
    palette = [
        "#000000", "#0718F5", "#963330", "#9133E7", "#ED9135", "#7DFBFD",
        "#74808E", "#FBEC98", "#000000", "#E73323", "#7DFA4C",
    ]
    clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    ax = pd.DataFrame(data=clf.transform(X_train)).plot.scatter(
        coords[0] - 1, coords[1] - 1, ax=ax, c='none', edgecolors=palette,
    )
    ax = pd.DataFrame(data=clf.transform(clf.means_)).plot.scatter(
        coords[0] - 1, coords[1] - 1, ax=ax, c='none', edgecolors=palette,
        linewidth=5,
    )
    ax.set_xlabel(f"Coordinate {coords[0]}")
    ax.set_ylabel(f"Coordinate {coords[1]}")
    ax.set_title("Linear Discriminant Analysis")
