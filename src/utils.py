import numpy as np
import pandas as pd

def sample_l2lap(eta:float, d:int) -> np.array:
  R = np.random.gamma(d, scale = 1.0 / eta)
  Z = np.random.normal(0, 1, size = d)
  return R * (Z / np.linalg.norm(Z)) #shape is (d,) one dimensional

def generate_synth_data(B_train, B_test, d):
    """
    Generates synthetic data for training and testing.

    Args:
        B_train (int): Number of training samples.
        B_test (int): Number of testing samples.
        d (int): Dimension of the feature vectors.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, b)
            X_train (numpy.ndarray): Training features, shape (B_train, d).
            y_train (numpy.ndarray): Training labels, shape (B_train,).
            X_test (numpy.ndarray): Testing features, shape (B_test, d).
            y_test (numpy.ndarray): Testing labels, shape (B_test,).
            b (numpy.ndarray) : noise vector of shape (d,)
    """
    # Generate random features with L2 norm <= 1
    X_train = np.random.randn(B_train, d)
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)

    X_test = np.random.randn(B_test, d)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    # Generate random labels (-1 or 1)
    y_train = np.random.choice([-1, 1], size=(B_train,))
    y_test = np.random.choice([-1, 1], size=(B_test,))

    return X_train, y_train, X_test, y_test