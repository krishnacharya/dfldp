
from src.utils import sample_l2lap, generate_synth_data
from src.LogLossAccuracy import LogLossAccuracy
from src.LPopt import LPOpt

class DFL:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_train, self.dim = X_train.shape
    
    def train_privately(self, epsilon:float, c:float, lamb:float):
        '''
            epsilon: privacy parameter, c is Q=cI, lamb is regularization parameter
            ---
            Pertubed Obj min_w \left[ -y_train^T (Q_inv (X_train w - G^T \gamma_dual)) \right] + 0.5 * lamb * ||w||^2 + 1/n * (b^T w)
            
            We assume Q = c I_n, so Q_inv = 1/c I_n

            Pertubed Obj has a closed form solution:
            theta_dpfl = 1/lamb (1/c * X_train^T y_train - 1/n * b)
        '''
        eta = c * epsilon / (4 * self.n_train**1.5)
        b = sample_l2lap(eta = eta, d = self.dim)
        self.w_pri = 1/lamb * (1/c * self.X_train.T @ self.y_train - 1/self.n_train * b)
        return self.w_pri

    def metrics_train(self, w, lpopt_train:LPOpt):
        '''
            Get logloss, accuracy, decision quality on training data

            lpopt is the optimization problem on the training data
        '''
        lla = LogLossAccuracy(self.X_train, self.y_train)
        ll = lla.get_logloss(w)
        acc = lla.get_accuracy(w)
        dq = lpopt_train.get_DQ(w)
        return ll, acc, dq

    def metrics_test(self, w, lpopt_test:LPOpt):
        '''
            Get logloss, accuracy, decision quality on test data

            lpopt is the optimization problem on the test data
        '''
        lla = LogLossAccuracy(self.X_test, self.y_test)
        ll = lla.get_logloss(w)
        acc = lla.get_accuracy(w)
        dq = lpopt_test.get_DQ(w)
        return ll, acc, dq

if __name__ == "__main__":
    # Example usage:
    B_train = 10000
    B_test = 10000
    d = 200

    X_train, y_train, X_test, y_test, b = generate_synth_data(B_train, B_test, d)
    lamb = 0.1
    c = 1
    model = DFL(X_train, y_train, X_test, y_test)
    w_pri = model.train_privately(epsilon=1000, c = c, lamb=lamb) # very large epsilon shouldnt make any difference, sanity check

    print(w_pri.shape)



        