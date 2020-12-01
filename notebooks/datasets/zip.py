import numpy as np

from pathlib import Path


X = [np.loadtxt(
    Path(__file__).absolute().parents[2] / "data" / "zip.digits" / f"train.{d}",
    delimiter=',',
).reshape(-1, 16, 16) for d in range(10)]
