
from utils import sample_l2lap, generate_synth_data

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



        