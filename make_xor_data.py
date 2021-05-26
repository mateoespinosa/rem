"""
Simple script to generate our XOR task dataset.
"""
import numpy as np
import pandas as pd

NUM_FEATS = 10
NUM_SAMPLES = 1000
DATA_OUTPUT_FILE = '../DNN-RE-data-new/XOR/data.csv'


def generate_xor(samples, features):
    """
    Generates the XOR dataset producing `samples` `features`-dimensional
    points.
    :param int samples: Number of total samples in XOR dataset.
    :param int features: Number of features used for a single sample of the
        dataset.
    """
    X = np.abs(np.random.rand(samples, features))
    y = np.logical_xor(np.round(X[:, 0]), np.round(X[:, 1]))
    return (X, y)


if __name__ == '__main__':
    X, y = generate_xor(samples=NUM_SAMPLES, features=NUM_FEATS)
    data = np.concatenate((X, np.expand_dims(y, axis=-1)), axis=-1)
    df = pd.DataFrame(
        data=data,
        columns=[f"feat_{i}" for i in range(NUM_FEATS)] + ["xor"],
    )
    df.to_csv(DATA_OUTPUT_FILE, index_label=False, index=False)
