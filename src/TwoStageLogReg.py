import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from src.utils import sample_l2lap, generate_synth_data
from src.LogLossAccuracy import LogLossAccuracy
from src.LPopt import L

class TwoStage:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_train = X_train.shape[0]
        self.dim = X_train.shape[1]

    def train_noprivacy(self, lamb):
        '''
            Train the logistic regression model without privacy.
            Uses L-BFGS-B optimization method on the Loss below

            Loss = 1/n * sum(log(1 + exp(-y_i * (X_i^T w)))) + 0.5 * lamb * ||w||^2
        '''
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
        result = minimize(logistic_loss, w_init, args=(self.X_train, self.y_train, lamb),
                          jac=logistic_loss_grad, method='L-BFGS-B')
        self.w_nopri = result.x
        return self.w_nopri

    def train_privately(self, epsilon, lamb):
        '''
            lamb is the regularization
            epsilon is the privacy parameter
            ----
            b is the noise vector sampled appropriately from Laplace
            Loss = \left[ 1/n * sum(log(1 + exp(-y_i * (X_i^T w)))) + 0.5 * lamb * ||w||^2 \right]  +  1/n * (b^T w) + 0.5 * \Delta * ||w||^2

        '''
        def logistic_loss_private(w, X, y, lamb, Delta, b):
            z = y * np.dot(X, w)
            w_norm_sq = np.linalg.norm(w)**2
            loss = np.mean(np.log(1 + np.exp(-z))) + 0.5 * lamb * w_norm_sq + np.dot(b, w)/self.n_train + 0.5 * Delta * w_norm_sq
            return loss

        def logistic_loss_private_grad(w, X, y, lamb, Delta, b):
            z = y * np.dot(X, w)
            sigmoid = 1 / (1 + np.exp(z))
            grad = np.mean(-(X * (y*sigmoid).reshape(-1,1)), axis=0) + b/self.n_train + (lamb+Delta) * w
            return grad
        
        def get_Delta_and_noisevec(epsilon, lamb):
            '''
                Get the Delta and the self.dim dimensional noise vector
            '''
            q = 1/4 # UB for second derivative of logloss
            epsilon_prime = epsilon - 2 * np.log(1 + 2*q / (self.n_train * lamb))
            if epsilon_prime > 0:
                Delta = 0
            else:
                Delta = (2*q) / (self.n_train*(np.exp(epsilon/4)-1)) - lamb
                epsilon_prime = epsilon / 2
            
            return Delta, sample_l2lap(eta=epsilon_prime/2, d=self.dim)

        Delta, b = get_Delta_and_noisevec(epsilon, lamb)
        w_init = np.random.randn(self.dim) * 0.01
        result = minimize(logistic_loss_private, w_init, args=(self.X_train, self.y_train, lamb, Delta, b),
                          jac=logistic_loss_private_grad, method='L-BFGS-B')
        self.w_pri = result.x
        return self.w_pri

    def metrics_train(self, w, G=None, h=None):
        '''
            Get logloss, accuracy, decision quality on training data

            G and h are the constraints in the LP formulation Gz \leq h, G has shape (m,n_test), h has shape (m,)
        '''
        pass

    def metrics_test(self, w, G=None, h=None):
        '''
            Get logloss, accuracy, decision quality on test data

            G and h are the constraints in the LP formulation Gz \leq h, G has shape (m,n_test), h has shape (m,)
        '''
        pass

if __name__ == "__main__":
    # Example usage:
    B_train = 10000
    B_test = 10000
    d = 200

    X_train, y_train, X_test, y_test, b = generate_synth_data(B_train, B_test, d)
    lamb = 0.1
    model = TwoStage(X_train, y_train, X_test, y_test)
    w_nopri = model.train_noprivacy(lamb=lamb)
    w_pri = model.train_privately(epsilon=1000, lamb=lamb) # very large epsilon shouldnt make any difference, sanity check

    print("Difference b/w nonpri and pri soln", np.linalg.norm(w_nopri - w_pri))
    print(w_pri.shape, w_nopri.shape)

    w_pri = model.train_privately(epsilon=1, lamb=lamb) #smaller epsilon should make a difference
    print("Difference b/w nonpri and pri soln", np.linalg.norm(w_nopri - w_pri))
    print(w_pri.shape, w_nopri.shape)


    # Train with scikit-learn
    sklearn_model = LogisticRegression(penalty='l2', C=1 / (lamb * X_train.shape[0]), fit_intercept=False, solver='lbfgs', max_iter=1000, tol=1e-6)
    sklearn_model.fit(X_train, y_train)
    w_nopri_sklearn = sklearn_model.coef_.flatten()

    # Compare the results
    # print("Our implementation w:", w_nopri)
    # print("Scikit-learn w:", w_nopri_sklearn)
    print("Difference nonpri and sklearn nonpri", np.linalg.norm(w_nopri - w_nopri_sklearn))
    