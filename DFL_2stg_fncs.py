import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


class TwoStage:
    def __init__(self, X_train, y_train, X_test, y_test, lamb):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lamb = lamb
        self.n_train = X_train.shape[0]
        self.dim = X_train.shape[1]

    def train_noprivacy(self):
        def logistic_loss(w, X, y, lamb):
            z = y * np.dot(X, w)
            loss = np.mean(np.log(1 + np.exp(-z))) + 0.5 * lamb * np.linalg.norm(w)**2
            return loss

        def logistic_loss_grad(w, X, y, lamb):
            z = y * np.dot(X, w)
            sigmoid = 1 / (1 + np.exp(z))
            grad = np.mean(-(X * (y*sigmoid).reshape(-1,1)), axis=0) + lamb * w
            return grad

        w_init = np.random.randn(self.dim) * 0.01
        result = minimize(logistic_loss, w_init, args=(self.X_train, self.y_train, self.lamb),
                          jac=logistic_loss_grad, method='L-BFGS-B')
        self.w_nopri = result.x
        return self.w_nopri

    def train_privately(self, b):
        def logistic_loss_private(w, X, y, lamb, b):
            z = y * np.dot(X, w)
            loss = np.mean(np.log(1 + np.exp(-z))) + np.dot(b, w)/self.n_train + 0.5 * lamb * np.linalg.norm(w)**2
            return loss

        def logistic_loss_private_grad(w, X, y, lamb, b):
            z = y * np.dot(X, w)
            sigmoid = 1 / (1 + np.exp(z))
            grad = np.mean(-(X * (y*sigmoid).reshape(-1,1)), axis=0) + b/self.n_train + lamb * w
            return grad

        w_init = np.random.randn(self.dim) * 0.01
        result = minimize(logistic_loss_private, w_init, args=(self.X_train, self.y_train, self.lamb, b),
                          jac=logistic_loss_private_grad, method='L-BFGS-B')
        self.w_pri = result.x
        return self.w_pri

def generate_dummy_data(B_train, B_test, d):
    """
    Generates dummy data for training and testing.

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

    #generate noise vector
    b = np.random.randn(d)

    return X_train, y_train, X_test, y_test, b


if __name__ == "__main__":
    # Example usage:
    B_train = 100
    B_test = 100
    d = 10

    X_train, y_train, X_test, y_test, b = generate_dummy_data(B_train, B_test, d)
    lamb = 0.1
    model = TwoStage(X_train, y_train, X_test, y_test, lamb)
    w_nopri = model.train_noprivacy()
    # print("Trained model without privacy:", w_nopri)
    # w_pri = model.train_privately(b)
    # print("Trained model with privacy:", w_pri)
    # Train with scikit-learn
    sklearn_model = LogisticRegression(penalty='l2', C=1 / (lamb * X_train.shape[0]), fit_intercept=False, solver='lbfgs', max_iter=1000, tol=1e-6)
    sklearn_model.fit(X_train, y_train)
    w_nopri_sklearn = sklearn_model.coef_.flatten()

    # Compare the results
    print("Our implementation w:", w_nopri)
    print("Scikit-learn w:", w_nopri_sklearn)
    print("Difference:", np.linalg.norm(w_nopri - w_nopri_sklearn))
    