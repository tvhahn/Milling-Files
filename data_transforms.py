import numpy as np
import h5py


def scaler(x, min_val, max_val, lower_norm_val, upper_norm_val):
    """Scale the signal between a min and max value
    
    Parameters
    ===========
    x : ndarray
        Signal that is being normalized

    max_val : int or float
        Maximum value of the signal or dataset

    min_val : int or float
        Minimum value of the signal or dataset

    lower_norm_val : int or float
        Lower value you want to normalize the data between (e.g. 0)

    upper_norm_val : int or float
        Upper value you want to normalize the data between (e.g. 1)

    Returns
    ===========
    x : ndarray
        Returns a new array that was been scaled between the upper_norm_val
        and lower_norm_val values

    """

    # https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range
    col, row = np.shape(x)
    for i in range(col):
        x[i] = np.interp(x[i], (min_val, max_val), (lower_norm_val, upper_norm_val))
    return x


def load_train_test(directory):

    path = directory

    with h5py.File(path / "X_train.hdf5", "r") as f:
        X_train = f["X_train"][:]
    with h5py.File(path / "y_train.hdf5", "r") as f:
        y_train = f["y_train"][:]

    with h5py.File(path / "X_train_slim.hdf5", "r") as f:
        X_train_slim = f["X_train_slim"][:]
    with h5py.File(path / "y_train_slim.hdf5", "r") as f:
        y_train_slim = f["y_train_slim"][:]

    with h5py.File(path / "X_val.hdf5", "r") as f:
        X_val = f["X_val"][:]
    with h5py.File(path / "y_val.hdf5", "r") as f:
        y_val = f["y_val"][:]

    with h5py.File(path / "X_val_slim.hdf5", "r") as f:
        X_val_slim = f["X_val_slim"][:]
    with h5py.File(path / "y_val_slim.hdf5", "r") as f:
        y_val_slim = f["y_val_slim"][:]

    with h5py.File(path / "X_test.hdf5", "r") as f:
        X_test = f["X_test"][:]
    with h5py.File(path / "y_test.hdf5", "r") as f:
        y_test = f["y_test"][:]

    return (
        X_train,
        y_train,
        X_train_slim,
        y_train_slim,
        X_val,
        y_val,
        X_val_slim,
        y_val_slim,
        X_test,
        y_test,
    )
